# OpenVDN 8 步 · ComfyUI · 8×H200

ComfyUI 负责参考图输入、任务队列和视频预览；推理调用固定版本的 [OpenVDN 官方仓库](https://github.com/OpenVDN/vdn-minimax-h3)。一条视频使用全部 8 张 H200，采用官方 H200 配置的 **6 个 softmax rank + 2 个 linear rank**，8 NFE，默认 FP8。

这里的参考图模式是官方 **Ref2VA-like**：使用 FL2VA 权重接收参考图，**不是 MiniMax 的独立 Ref2VA transformer**。当前仅接收图片参考，不接收参考音频或参考视频。LightX2V 已从当前部署方案移除。

这是官方后端的简单封装。原先要求的 Sol、跨步 DiT 缓存、整块 DiT 编译没有接入：VDN 已有自己的混合注意力，官方没有提供这三个可直接复用的开关。界面的 `inference_kernels` 是官方融合/编译内核组合开关，不能当作整块 DiT 的 `torch.compile` 开关。

## 在 8×H200 服务器部署

环境：Linux x86_64，完整的 8×H200，能运行 CUDA 12.9 的 NVIDIA 驱动和可用的 NVLink/NCCL。安装需要 Git、curl、CA 证书和 C/C++ 编译工具。为模型、两个 Python 环境和编译缓存预留约 250 GB 空间。模型约 140 GB，其中 Qwen3-VL 条件编码器约 62 GB。

在服务器拉取本仓库后，进入仓库目录，只需执行一个命令：

```bash
bash deploy.sh
```

默认依次完成环境安装、权重下载、八卡 NCCL 检查和 ComfyUI 启动；任一步失败都会停止，不会继续启动不完整的服务。重复运行会复用下载与安装缓存。`install`、`download`、`check`、`start` 子命令仍可用于单独排查。

浏览器访问 `http://服务器IP:8188`，在工作流列表打开 `openvdn_ref2va_like.json`（首次启动自动放入列表，后续保留你的修改）。在 Load Image 节点上传自己的参考图，编辑提示词后点击运行，生成节点会显示带音频的视频。增加参考图时复制 Load Image + OpenVDN Reference 两个节点，并用 `previous` 串联 Reference 节点；顺序对应 `<Picture 1>`、`<Picture 2>`……本封装最多 9 张。

命令在前台运行。可以放在 tmux 中；生成过程中先点 ComfyUI 的取消按钮，再退出服务。取消会终止当前编码进程或整组 torchrun 子进程。

安装器固定 Python 3.12，并建立两个环境：

| 环境 | 用途 | 主要版本 |
| --- | --- | --- |
| `.venv-ui` | ComfyUI，CPU 模式 | ComfyUI 0.30.0、torch 2.10、Transformers 4.57.6 |
| `.venv-vdn` | 官方编码和八卡推理 | torch 2.13.0+cu129、Transformers 5.15、FlashAttention 4 |

源码/模型的精确 revision 见 `sources.lock.json`。Diffusers 固定在官方指定的 base，并应用该 VDN 版本自带的全部补丁。重复安装不会重复打补丁；遇到本地修改会报错并保留文件。其余依赖按官方 `pyproject.toml` / ComfyUI requirements 安装，并非所有间接依赖都完全锁定。

可在上述命令前设置：

```bash
export REF2VA_MODELS=/data/models/openvdn-h3  # 默认 ./models；download/start/render 使用同一个值
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export REF2VA_PORT=8188
export REF2VA_LISTEN=0.0.0.0
# 如下载需要账号：export HF_TOKEN=你的HuggingFaceToken
```

推理进程使用本地权重并开启 Hub 离线模式，不会在生成过程中下载另一版模型。

## 控制项

| 控制项 | 默认 | 含义 |
| --- | --- | --- |
| `task` | `ref2va_like` | 图片参考；可改为 `t2va` 并断开 references |
| `num_frames` | 345 | 24 fps，约 14.4 秒；支持 107–345 的 `17n+5` 帧数 |
| `seed` | 42 | 比较配置时固定 seed |
| `reference_short_edge` | 768 | 每张参考图的短边；提高到 2048 会明显增加参考 token 和耗时 |
| `fp8` | true | 官方 FP8 线性层；关闭使用 BF16，可能改变速度、显存及结果 |
| `inference_kernels` | true | 官方融合和局部编译内核；关闭也不代表 Flex/Triton 不编译 |
| `softmax_backend` | flex | 官方 H200 默认；可比较 decomposed / ref |
| `softmax_ranks` | 6 | 6 softmax + 2 linear；0 切换为普通 Ulysses，仍使用 8 卡 |
| `warmup_steps` | 2 | 计时前额外执行并丢弃的预热 NFE；正式采样仍是 8 NFE |
| `profile` | false | 开启各 rank 的 CUDA 分段计时；性能对比时通常关闭 |

输出画布沿用官方固定的 **1344×768**。保持官方训练时的 video/audio shift=12/3，不开放无对应权重的步数或 shift 调节。八卡通信、注意力、采样和解码代码均直接使用官方实现。

## 首次验证与计时

先用官方已编码的参考图示例，跳过 Qwen3-VL 编码，验证八卡后端：

```bash
./deploy.sh render \
  --prompt-file .deps/openvdn/prompts/reference/example_ref2va.pt \
  --output output/official_ref2va_like.mp4
```

自己的提示词和参考图也可从命令行测试，调用的后端与 ComfyUI 相同：

```bash
./deploy.sh render \
  --prompt 'The person in <Picture 1> walks toward the camera and waves. Natural ambient sound.' \
  --refs input/person.png \
  --seed 42 --num-frames 345 \
  --output output/my_ref2va_like.mp4
```

查看 `./deploy.sh render --help` 获得其他参数。CLI 使用 `--no-fp8` 和 `--no-inference-kernels` 关闭对应控制。输出文件已存在会拒绝覆盖。

每条任务都执行以下流程：

1. 对 prompt、参考图内容及模型版本计算缓存键。命中则复用条件张量；未命中则在 GPU 0 加载 Qwen3-VL 并编码，进程退出后释放显存。
2. 启动官方八卡 `infer_ulysses.py`，加载模型、预热、执行正式 8 NFE；rank 0 解码视频与音频并写 MP4。
3. 保存官方原始计时和封装层计时，退出整个八卡进程组。

**每条任务都会重新加载 DiT**。磁盘上的编译缓存可复用，但这不是模型常驻服务；第一次编译可能需要数分钟。这里的“条件缓存”和“编译缓存”都不是跨去噪步的 DiT 结果缓存。同一目录的 UI/CLI 通过文件锁串行使用八卡；不要从多个部署副本同时占用同一组卡。

产物：

- ComfyUI 视频：`output/openvdn/*.mp4`
- 官方计时和内核实际状态：`视频.mp4.inference.json`
- 完整任务记录：`视频.mp4.metrics.json`，含 queue/encode/inference process/request wall time，以及官方 `upstream.timings`
- 命令、配置和错误日志：`.runtime/jobs/<job_id>/`
- 条件与编译缓存：`.runtime/conditioning/`、`.runtime/inductor/`、`.runtime/triton/`
- ComfyUI 数据库：`.runtime/comfy-user/comfyui.db`；启动脚本显式设置路径并创建父目录，不依赖 ComfyUI 源码中的默认 `user/` 目录。

官方报告的 H200 **18.3 秒**指 8 NFE 的去噪耗时，并非包含编码器、模型加载、预热和 MP4 编码的整条请求耗时，也不是这里测出的结果。与它对比应查看 `upstream.timings.denoise_seconds`；实际使用等待时间看 `request_wall_seconds`。参考图数量和大小也会影响去噪时间。[官方结果和说明](https://github.com/OpenVDN/vdn-minimax-h3#results)

## 验证范围

本地已验证 Python 请求/缓存/进程取消逻辑、官方 Diffusers 补丁可应用且可重复安装，以及固定版本 ComfyUI 在 CPU 环境中能注册这两个节点。没有在本机安装 Linux CUDA 环境、下载大模型或运行 H200 推理；画质、GPU 峰值显存及速度需要服务器实测。

`./deploy.sh check --nccl` 会检查 8 张完整 H200、固定版本环境/模型、官方配置加载，以及真实的八卡 NCCL all-reduce/all-to-all。通过后再运行上述官方示例。源码、权重的授权条件见 [OpenVDN LICENSE](https://github.com/OpenVDN/vdn-minimax-h3#license) 和模型仓库说明。
