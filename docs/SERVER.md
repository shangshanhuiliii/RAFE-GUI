# Linux 服务器运行

## 首次部署

在服务器上克隆代码。以后在这个目录内更新。

```bash
cd ~/qizheng.hao/guiagent
git clone https://github.com/shangshanhuiliii/RAFE-GUI.git rafe-code
cd rafe-code
```

将已解压的 RAFE-GUI 传输包中的 `data/` 复制到克隆目录。两套模型权重分别放在 `models/encoders/siglip-so400m-patch14-384/` 和 `models/encoders/dinov2-giant/`，每套包括权重、模型配置和图像处理器配置。数据和权重可从服务器已有目录复制或链接。

进入已有 CUDA PyTorch 的 Python 3.11 或以上环境，安装项目和评分依赖。

```bash
conda activate guiagent
python -m pip install -e ".[encoders,analysis]"
```

## 更新代码

在克隆目录执行以下命令。数据、模型权重和实验输出由服务器本地管理。依赖清单有更新时，重新执行上面的安装命令。

```bash
git pull --ff-only
```

## 启动

在 Linux 和 PowerShell 中使用同一条命令。

```bash
python -m rafe_gui run
```

每次启动生成独立运行 ID，先跑 `square-pad`，再跑 `stretch`。每种输入方式用 GPU 0 跑 SigLIP、GPU 1 跑 DINOv2。终端会立即显示运行 ID 和输出位置，随后显示各阶段进度。

```text
[RAFE-GUI] Checking dataset and prepared images
[RAFE-GUI] 1/2 square-pad: preparing inputs
[square-pad | siglip] 10/800 texture ok (0.20s)
[square-pad | dino] 10/800 texture ok (0.35s)
```

命令在当前终端前台运行。两组完成后，终端会打印 `Completed` 和对比报告路径。报告位于 `runs/RUN_ID/comparison/report.md`，各组的结果在相应的 `analysis/` 内。

使用 `Ctrl+C` 停止。需要继续本次运行时，将 `RUN_ID` 替换为启动时打印的运行 ID。

```bash
python -m rafe_gui run --run-id RUN_ID --resume
```
