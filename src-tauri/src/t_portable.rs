use serde::{Deserialize, Serialize};
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tauri::{PhysicalPosition, PhysicalSize, WebviewWindow, Window};

static DATA_DIR: OnceLock<PathBuf> = OnceLock::new();

pub fn initialize() -> Result<(), String> {
    let mut executable_dir = std::env::current_exe().map_err(|error| error.to_string())?;
    executable_dir.pop();
    let settings_file = executable_dir.join("lap-data-dir.txt");
    let value = match fs::read_to_string(&settings_file) {
        Ok(value) => value,
        Err(error) if error.kind() == ErrorKind::NotFound => String::new(),
        Err(error) => return Err(format!("Cannot read {}: {error}", settings_file.display())),
    };
    let value = value.trim_start_matches('\u{feff}').trim();
    let data = executable_dir.join(if value.is_empty() { "data" } else { value });
    if !data.is_absolute() {
        return Err(format!("Use an absolute path or a path relative to Lap in {}", settings_file.display()));
    }
    let temp = data.join("tmp");
    let webview = data.join("webview");
    for path in [&temp, &webview, &data.join("cache")] {
        fs::create_dir_all(path).map_err(|error| format!("Cannot create {}: {error}", path.display()))?;
    }
    // Windows synchronizes environment access. Set the paths before creating any WebView.
    unsafe {
        std::env::set_var("WEBVIEW2_USER_DATA_FOLDER", webview);
        std::env::set_var("TMP", &temp);
        std::env::set_var("TEMP", temp);
    }
    DATA_DIR.set(data).expect("portable paths are initialized once at startup");
    Ok(())
}

pub fn data_dir() -> &'static Path {
    DATA_DIR.get().expect("portable paths initialized before app setup")
}

pub fn show_startup_error(message: &str) {
    use windows::{
        core::{PCWSTR, w},
        Win32::UI::WindowsAndMessaging::{MB_ICONERROR, MB_OK, MessageBoxW},
    };
    let message = message.encode_utf16().chain(Some(0)).collect::<Vec<_>>();
    unsafe {
        MessageBoxW(None, PCWSTR(message.as_ptr()), w!("Lap NPU"), MB_OK | MB_ICONERROR);
    }
}

#[derive(Serialize, Deserialize)]
struct MainWindowState {
    width: u32,
    height: u32,
    x: i32,
    y: i32,
    maximized: bool,
}

pub fn restore_main_window(window: &WebviewWindow) {
    let Ok(file) = fs::File::open(data_dir().join("window-state.json")) else {
        return;
    };
    let Ok(state) = serde_json::from_reader::<_, MainWindowState>(file) else {
        return;
    };
    if state.maximized {
        let _ = window.maximize();
    } else {
        let _ = window.set_size(PhysicalSize::new(state.width, state.height));
        if window.monitor_from_point(f64::from(state.x), f64::from(state.y)).ok().flatten().is_some() {
            let _ = window.set_position(PhysicalPosition::new(state.x, state.y));
        }
    }
}

pub fn save_main_window(window: &Window) -> Result<(), Box<dyn std::error::Error>> {
    if window.is_minimized()? {
        return Ok(());
    }
    let size = window.inner_size()?;
    let position = window.outer_position()?;
    let state = MainWindowState {
        width: size.width,
        height: size.height,
        x: position.x,
        y: position.y,
        maximized: window.is_maximized()?,
    };
    fs::write(data_dir().join("window-state.json"), serde_json::to_vec_pretty(&state)?)?;
    Ok(())
}
