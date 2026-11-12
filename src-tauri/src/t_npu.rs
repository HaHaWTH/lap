//! OpenVINO encoders for the Windows x64 NPU build.
use openvino::{
    Core, DeviceType, ElementType, InferRequest, PartialShape, PropertyKey, RwPropertyKey, Shape,
    Tensor,
};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const IMAGE_SHAPE: [i64; 4] = [1, 3, 224, 224];
const EMBEDDING_SIZE: usize = 512;

fn executable_dir() -> Result<PathBuf> {
    let mut executable = std::env::current_exe()?;
    executable.pop();
    Ok(executable)
}

fn core_for(device: &DeviceType<'_>) -> Result<Core> {
    openvino_sys::library::load_from(executable_dir()?.join("openvino_c.dll"))?;
    let mut core = Core::new()?;
    if device == &DeviceType::NPU {
        core.set_property(
            device,
            &RwPropertyKey::Other("NPU_BYPASS_UMD_CACHE".into()),
            "YES",
        )?;
    }
    Ok(core)
}

fn read_model(core: &mut Core, model_path: &Path) -> Result<openvino::Model> {
    let model_path = model_path.to_str().ok_or("Model path is not valid Unicode")?;
    Ok(core.read_model_from_file(model_path, "")?)
}

fn request_for(
    core: &mut Core,
    model: &openvino::Model,
    device: DeviceType<'_>,
) -> Result<(InferRequest, String)> {
    let mut compiled = core.compile_model(model, device)?;
    let devices = compiled
        .get_property(&PropertyKey::Other("EXECUTION_DEVICES".into()))?
        .into_owned();
    Ok((compiled.create_infer_request()?, devices))
}

fn run_embedding(request: &mut InferRequest, output_name: &str) -> Result<Vec<f32>> {
    request.infer()?;
    let tensor = request.get_tensor(output_name)?;
    if tensor.get_element_type()? != ElementType::F32 || tensor.get_size()? != EMBEDDING_SIZE {
        return Err(format!(
            "Unexpected OpenVINO output for {output_name}: expected 512 float32 values"
        )
        .into());
    }
    Ok(tensor.get_data::<f32>()?.to_vec())
}

pub struct VisionEncoder {
    request: InferRequest,
    pixels: Tensor,
    execution_devices: String,
}

impl VisionEncoder {
    pub fn open(model_path: &Path, device: DeviceType<'_>) -> Result<Self> {
        let mut core = core_for(&device)?;
        let mut model = read_model(&mut core, model_path)?;
        model.reshape_single_input(&PartialShape::new_static(4, &IMAGE_SHAPE)?)?;
        let (mut request, execution_devices) = request_for(&mut core, &model, device)?;
        let pixels = Tensor::new(ElementType::F32, &Shape::new(&IMAGE_SHAPE)?)?;
        request.set_tensor("pixel_values", &pixels)?;
        Ok(Self {
            request,
            pixels,
            execution_devices,
        })
    }

    pub fn encode(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        self.pixels.get_data_mut::<f32>()?.copy_from_slice(input);
        run_embedding(&mut self.request, "image_embeds")
    }
}

pub struct TextEncoder {
    request: InferRequest,
    input_ids: Tensor,
    attention_mask: Option<Tensor>,
    pad_id: i64,
    output_name: &'static str,
    execution_devices: String,
}

impl TextEncoder {
    pub fn open(
        model_path: &Path,
        tokenizer: &Tokenizer,
        multilingual: bool,
        device: DeviceType<'_>,
    ) -> Result<Self> {
        let (length, pad_token, output_name) = if multilingual {
            (512, "[PAD]", "sentence_embeddings")
        } else {
            (77, "<|endoftext|>", "text_embeds")
        };
        let pad_id = tokenizer
            .token_to_id(pad_token)
            .ok_or_else(|| format!("Tokenizer is missing {pad_token}"))?;
        let dimensions = [1, length];
        let shape = Shape::new(&dimensions)?;
        let partial_shape = PartialShape::new_static(2, &dimensions)?;
        let mut core = core_for(&device)?;
        let mut model = read_model(&mut core, model_path)?;
        if multilingual {
            model.reshape(&[
                ("input_ids", &partial_shape),
                ("attention_mask", &partial_shape),
            ])?;
        } else {
            model.reshape_single_input(&partial_shape)?;
        }
        let (mut request, execution_devices) = request_for(&mut core, &model, device)?;
        let input_ids = Tensor::new(ElementType::I64, &shape)?;
        request.set_tensor("input_ids", &input_ids)?;
        let attention_mask = if multilingual {
            let mask = Tensor::new(ElementType::I64, &shape)?;
            request.set_tensor("attention_mask", &mask)?;
            Some(mask)
        } else {
            None
        };
        Ok(Self {
            request,
            input_ids,
            attention_mask,
            pad_id: i64::from(pad_id),
            output_name,
            execution_devices,
        })
    }

    pub fn encode(&mut self, tokens: &[u32], mask: &[u32]) -> Result<Vec<f32>> {
        let ids = self.input_ids.get_data_mut::<i64>()?;
        if tokens.len() > ids.len() {
            return Err(format!("Search query exceeds the model's {} token limit", ids.len()).into());
        }
        for (slot, &id) in ids.iter_mut().zip(tokens) {
            *slot = i64::from(id);
        }
        ids[tokens.len()..].fill(self.pad_id);
        if let Some(attention) = self.attention_mask.as_mut() {
            let data = attention.get_data_mut::<i64>()?;
            for (slot, &bit) in data.iter_mut().zip(mask) {
                *slot = i64::from(bit);
            }
            data[mask.len()..].fill(0);
        }
        run_embedding(&mut self.request, self.output_name)
    }
}

fn check_embedding(values: &[f32]) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) || values.iter().all(|&value| value == 0.0) {
        return Err("OpenVINO returned an invalid embedding".into());
    }
    Ok(())
}

fn check_text(path: &Path, multilingual: bool, device: DeviceType<'_>) -> Result<serde_json::Value> {
    let tokenizer =
        Tokenizer::from_file(path.join("tokenizer.json")).map_err(|error| error.to_string())?;
    let mut model = TextEncoder::open(&path.join("text_model.onnx"), &tokenizer, multilingual, device)?;
    let prompt = if multilingual { "海边日落" } else { "sunset at the beach" };
    let tokens = tokenizer
        .encode(prompt, true)
        .map_err(|error| error.to_string())?;
    let vector = model.encode(tokens.get_ids(), tokens.get_attention_mask())?;
    check_embedding(&vector)?;
    Ok(serde_json::json!({
        "execution_devices": model.execution_devices,
        "dimensions": vector.len(),
    }))
}

fn verify(device: DeviceType<'_>, multilingual: Option<PathBuf>) -> Result<serde_json::Value> {
    #[cfg(feature = "portable-data")]
    crate::t_portable::initialize()?;

    let models = executable_dir()?.join("models");
    let mut image = VisionEncoder::open(&models.join("vision_model.onnx"), device.to_owned())?;
    let pixels = vec![0.0; IMAGE_SHAPE.iter().product::<i64>() as usize];
    check_embedding(&image.encode(&pixels)?)?;
    let started = Instant::now();
    for _ in 0..5 {
        check_embedding(&image.encode(&pixels)?)?;
    }
    let vision_ms = started.elapsed().as_secs_f64() * 200.0;
    let mut report = serde_json::json!({
        "vision": { "execution_devices": image.execution_devices, "dimensions": EMBEDDING_SIZE, "mean_ms": vision_ms },
        "english": check_text(&models, false, device.to_owned())?,
    });
    #[cfg(feature = "portable-data")]
    {
        report["data_dir"] = serde_json::json!(crate::t_portable::data_dir());
        report["temp_dir"] = serde_json::json!(std::env::temp_dir());
        report["webview_dir"] = serde_json::json!(std::env::var("WEBVIEW2_USER_DATA_FOLDER")?);
    }
    if let Some(path) = multilingual {
        report["multilingual"] = check_text(&path, true, device)?;
    }
    Ok(report)
}

// A headless entry point for checking the distributed files without opening a photo library.
pub fn run_check_if_requested() {
    let mut args = std::env::args_os().skip(1);
    let device = match args.next().as_deref() {
        Some(arg) if arg == OsStr::new("--check-npu") => DeviceType::NPU,
        Some(arg) if arg == OsStr::new("--check-openvino-cpu") => DeviceType::CPU,
        _ => return,
    };
    let Some(report_path) = args.next() else {
        std::process::exit(2);
    };
    let (report, status) = match verify(device, args.next().map(PathBuf::from)) {
        Ok(report) => (report, 0),
        Err(error) => (serde_json::json!({ "error": error.to_string() }), 1),
    };
    let report = serde_json::to_vec_pretty(&report).expect("a JSON value is serializable");
    std::process::exit(if std::fs::write(report_path, report).is_ok() {
        status
    } else {
        2
    });
}
