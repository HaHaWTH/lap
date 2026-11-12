# Lap NPU 便携版

解压整个 ZIP，双击 `Lap-NPU.exe`。这个版本用 Intel AI Boost NPU 生成图片和搜索文字的向量。安装包内的英文模型可以直接使用，中文搜索在「设置 → 搜索 → 选择模型」下载并启用「多语言」。

默认把应用数据放在程序旁的 `data` 目录。退出 Lap 后，编辑同级的 `lap-data-dir.txt`，写入一行绝对路径（例如 `E:\LapData`）或相对路径（例如 `data`）。下次启动从该目录读取配置和图库索引。需要保留已有图库时，先把旧数据目录里的内容复制到新目录。

应用配置在 `app-config.json`，数据库默认在 `libraries`，多语言模型在 `models`，缩略图和视频缓存放在 `cache`，浏览器组件的用户数据和临时文件分别在 `webview` 和 `tmp`。应用里另外指定的数据库目录优先于默认的 `libraries`。

Windows x64 便携包由 GitHub Actions 的 **Build Lap NPU Portable** 工作流生成。源代码基于 Lap v0.3.3，模型运行时为 OpenVINO 2025.1.0。
