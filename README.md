# RAFE-GUI

独立的 Real–Sim GUI 视觉保真度实验工程。数据、预处理图片和模型均从本项目内读取。

## 从哪里开始

服务器使用 Git 克隆和更新代码，数据与模型放在服务器本地。首次部署见[服务器说明](docs/SERVER.md)。安装后，Linux 和 PowerShell 都使用同一条命令。

```bash
python -m rafe_gui run
```

每次启动会创建一个带时间戳的运行目录。当前终端依次显示输入校验、权重加载、两个模型的进度和报告位置。也可以使用 `rafe run`。默认读取 `configs/text_fidelity.json`，其中的数据、模型和输出路径均相对于项目根目录解析。

默认配置面向服务器双GPU：GPU0运行SigLIP，GPU1运行DINOv2；先补白，再直接缩放。原图+四种文字处理、两个模型、两种几何共20套分数。当前A/B分别有204/168个ready配对，Eleme两账号直接对照17/13对，微信16/16对。float32不自动改为float16。

本机只有一张GPU时，可复制配置并将两项devices都设为cpu，或仅选择一个模型进行单独实验。默认双卡配置不适合直接在本机启动全量任务。

## 文件放置

```text
configs/             实验JSON、OCR配置
src/rafe_gui/        可维护的正式代码
data/dataset/        原图、metadata.json、eval_pairs与A/B方案
data/prepared/       已校验的共享OCR框、五条件图片、gallery.html
models/encoders/     SigLIP、DINOv2
models/ocr/          EasyOCR、PP-OCRv5检测权重
runs/<run-id>/       本次运行的状态、特征和报告
third_party/         GUI-CC许可证、来源、原始协议说明
```

真实环境的截图存放在 `data/dataset/Real/` 下。

| 目录 | 采集环境 |
|---|---|
| `WeChat/wechat_1` | 微信账号一（默认配置） |
| `WeChat/wechat_2` | 微信账号二（个性化配置） |
| `Eleme/eleme_app_1` | 淘宝闪购独立 APP 账号一 |
| `Eleme/eleme_app_2` | 淘宝闪购独立 APP 账号二 |
| `Eleme/eleme_miniapp` | 淘宝闪购微信小程序 |

代码模块按 `data`、`transforms`、`metrics`、`execution`、`analysis` 和 `storage` 分工。参数和模型路径不写死在这些模块中。扩展说明见[架构说明](docs/ARCHITECTURE.md)。

## 图像与评价协议

- `square-pad`：1260×2800整页等比缩至252×560，左右补白至560×560。
- `stretch`：1260×2800整页直接缩放560×560，改变宽高比，不裁剪。
- `rectangle`：整页等比缩至252×560，不补白。
- `baseline`：保留原GUI-CC的checkpoint处理器默认行为，包括DINOv2默认裁剪。

四种文字处理使用同一OCR框和Mask：固定灰色、局部背景、高斯模糊、确定性无语义伪笔画。已有160张预处理图片可直接复用，prepare只做验证；无需安装OCR依赖。对新数据运行OCR时，在独立环境安装 `.[ocr]`；EasyOCR另选 `.[easyocr]`。不自动下载任何模型。

`configs/vlm_strict.json`保留原GUI-CC的严格单对图片提示词、JPEG编码和分数归一化。`configs/vlm_grouped.json`为RAFE逻辑页面组扩展。两者分开记录，均默认使用原始截图，不套用编码器的560×560缩放。

## 结果、续跑与API

```bash
python -m rafe_gui status --run-id RUN_ID
python -m rafe_gui analyze --run-id RUN_ID
python -m rafe_gui run --run-id RUN_ID --resume
```

以终端打印的运行 ID 替换 `RUN_ID`。两种输入方式的对比报告在 `runs/RUN_ID/comparison/report.md`。终端打印 `Completed` 即表示评分和汇总全部结束。`Ctrl+C` 可中断本次运行，使用同一 ID 和 `--resume` 可继续。

VLM配置：将 `.env.example` 复制为 `.env`，填写 `RAFE_API_KEY`、`RAFE_API_BASE_URL`、`RAFE_JUDGE_MODEL`；模型名必须显式指定。安装 `.[vlm]` 后执行：

```bash
python -m rafe_gui run --config configs/vlm_strict.json --run-id strict_v1
```

该命令会调用配置中的 VLM API。
