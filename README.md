# OpenVDN 8 步 · ComfyUI · 8×H200

一条视频使用全部 8 张 H200。ComfyUI 负责输入、队列和视频预览；常驻八卡进程调用固定版本的 [OpenVDN](https://github.com/OpenVDN/vdn-minimax-h3)，默认 FP8、6 个 softmax rank + 2 个 linear rank、8 NFE。

图片参考模式是官方 **Ref2VA-like**：FL2VA 权重接收参考图，不是 MiniMax 的独立 Ref2VA transformer。暂不接入 LightX2V、Sol、跨步 DiT 缓存或整块 DiT 编译；`inference_kernels` 控制官方融合/局部编译内核组合。

## 一条命令启动

已部署的服务器更新：

```bash
cd /root/ref2va && git pull --ff-only && bash deploy.sh start
```

首次部署，在克隆的仓库目录执行 `bash deploy.sh`，自动安装固定版本源码、下载模型并启动。需要 Linux x86_64、8 张完整 H200、支持 CUDA 12.9 的驱动、NVLink/NCCL，约 250 GB 磁盘空间。命令在前台运行，可放入 tmux。

启动依次执行：

1. 停止本目录的旧服务。
2. **自动停止所选 8 张 GPU 上已有的计算应用**，释放显存。识别到 systemd 应用服务或 Docker 容器时停止其服务/容器；其他情况停止推理进程树，先 TERM，再在超时后 KILL。不会卸载应用或永久禁用服务。清理动作写入 `.runtime/gpu-cleanup.json`。若外部调度器持续拉起应用，启动报错，不会无限杀进程。
3. 检查模型、八卡环境与 NCCL。
4. 加载八份 DiT 和 rank 0 的视频/音频 VAE；Qwen3-VL 条件编码器通过 Accelerate 分配到这 8 张 GPU，单卡权重预算 12 GiB，禁止 CPU/磁盘权重卸载。
5. 使用一张合成参考图完成条件编码、正式 8 NFE、音视频解码及 MP4 编码预热。默认预热 10 秒、9:16、短边 768。
6. 预热成功后才启动 ComfyUI，访问 `http://服务器IP:8188`。

这是专用八卡服务的启动行为，会中止这些卡上原有的生成任务。设置 `REF2VA_CLEAR_GPU_APPS=0` 可关闭自动清理，改为显存不足时直接退出。清理只在启动执行；生成期间不会停止其他应用。

模型全程常驻。后续生成复用 DiT、Qwen3-VL 和 VAE，不重新读权重，不执行额外去噪预热。同一 prompt、参考图内容、参考尺寸和模型版本命中条件缓存时，也会跳过编码。新序列长度/尺寸仍可能触发内核编译；启动预热不能覆盖所有输入形状。保留磁盘编译缓存，并在有限数量的新形状后重置 Dynamo 编译图记录，避免官方单次推理实现达到重编译上限。

Ctrl-C 会回收 UI 和整个八卡进程组。取消正在推理的任务会终止整组 NCCL worker，并自动重新加载预热，期间生成接口返回 503。GPU/OOM 等非取消错误会保留 UI 和诊断接口，需执行上述启动命令恢复。

## JSON 接口

提交：`POST /openvdn/jobs`。返回 HTTP 202、`job_id`、`status_url` 和实际输出规格；通过 `GET /openvdn/jobs/{job_id}` 查询。该接口与 UI、CLI 共用队列/文件锁，一次只生成一条视频。

```bash
curl -sS http://43.218.119.131:8188/openvdn/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "The person in <Picture 1> walks toward the camera and waves. Natural ambient sound.",
    "duration": 10,
    "ratio": "9:16",
    "resolution": 720,
    "reference_image_urls": ["https://example.com/person.png"],
    "seed": 42
  }'
```

| 参数 | 含义 |
| --- | --- |
| `prompt` | 必填，最多 24000 字符；图片顺序对应 `<Picture 1>`、`<Picture 2>`…… |
| `duration` | 秒，4–15，默认 5；按 24 fps 四舍五入到整数帧 |
| `ratio` | 宽:高，如 `16:9`、`9:16`、`1:1`，默认 `16:9`；范围 1:4–4:1 |
| `resolution` | 输出短边像素，256–1080 的偶数，默认 720；长边四舍五入到偶数 |
| `reference_image_urls` | 必填，1–9 个 HTTP(S) 公网图片 URL，最多 20 MiB/张；不接收内网、文件或带凭据的 URL |
| `reference_image_url` | 单张图片的简写；与复数参数二选一 |
| `seed` | 默认 42 |
| `reference_short_edge` | 编码参考图的短边，默认 768、32 的倍数；越大越耗时，独立于输出 `resolution` |

例如 `duration=10, ratio=9:16, resolution=720` 输出 **720×1280、240 帧、10 秒**。模型内部在 736×1280、243 帧上生成，再缩放到输出尺寸并裁到目标时长，音频同步裁剪。内部宽高对齐 32、帧数对齐 `17n+5`。内部画布面积不超过 1920×1088；支持参数范围不代表所有高分辨率、长时长、多参考组合都能装入显存。

成功状态包含 `video_url`（以 `/view?...` 开头，相对于服务地址）和完整 `metrics`。失败包含错误详情。状态保存在磁盘，服务器重启后未完成任务标为 interrupted；不会自动重跑。

`GET /openvdn/health` 返回 `ready`、当前阶段、八卡信息和常驻模型的 `profile`。就绪/忙碌返回 200，失败或重新预热返回 503。POST 在后端未就绪时返回 503；模型配置与常驻配置不符返回 400。

## ComfyUI 与 CLI

工作流列表包含：

- `openvdn_url_request.json`：填写时长、比例、短边、参考图片 URL。
- `openvdn_ref2va_like.json`：兼容原工作流，通过 Load Image 上传图片，Reference 节点可串联；原节点仍用 1344×768 和 `num_frames`。可选择 `t2va` 并断开参考图。

首次启动放入工作流列表，后续不会覆盖用户保存的修改。结果节点预览带声音的视频。

服务启动后，CLI 复用同一常驻后端：

```bash
bash deploy.sh render \
  --prompt 'The person in <Picture 1> walks toward the camera and waves. Natural ambient sound.' \
  --refs input/person.png --duration 10 --ratio 9:16 --resolution 720 \
  --output output/my_ref2va_like.mp4
```

`--prompt-file` 支持复用官方 `.pt` 条件缓存；不能与 prompt/refs 同传。输出已存在时拒绝覆盖。

## 启动配置

| 环境变量 | 默认 | 含义 |
| --- | --- | --- |
| `REF2VA_CLEAR_GPU_APPS` | 1 | 启动时清理所选 GPU 的计算应用；0 仅检查 |
| `CUDA_VISIBLE_DEVICES` | 0,1,2,3,4,5,6,7 | 恰好 8 张卡，可用 GPU UUID |
| `REF2VA_MODELS` | ./models | 下载、启动使用同一权重目录 |
| `REF2VA_PORT` / `REF2VA_LISTEN` | 8188 / 0.0.0.0 | ComfyUI 地址 |
| `REF2VA_FP8` | 1 | 官方 FP8 线性层；0 为 BF16 |
| `REF2VA_INFERENCE_KERNELS` | 1 | 官方融合/局部编译内核；0 也不代表全部禁用编译 |
| `REF2VA_SOFTMAX_BACKEND` | flex | flex / decomposed / ref |
| `REF2VA_SOFTMAX_RANKS` | 6 | 6+2；0 为普通八卡 Ulysses |
| `REF2VA_PROFILE` | 0 | 各 rank 分段计时 |
| `REF2VA_WARMUP_DURATION` | 10 | 启动预热时长 |
| `REF2VA_WARMUP_RATIO` | 9:16 | 启动预热画幅 |
| `REF2VA_WARMUP_RESOLUTION` | 768 | 启动预热短边 |
| `REF2VA_X264_PRESET` | veryfast | CPU H.264 编码预设，可改 medium；保持 CRF 23，更快预设可能增大文件并改变压缩细节 |
| `REF2VA_X264_THREADS` | 8 | H.264 编码线程数，1–64 |
| `REF2VA_REFERENCE_SHORT_EDGE` | 768 | 启动预热参考图短边 |

要比较精度/内核/并行配置，修改相应环境变量再执行 `bash deploy.sh start`。API 省略这些字段时自动使用当前配置；显式传入 `fp8`、`inference_kernels`、`softmax_backend`、`softmax_ranks`、`profile` 必须与 `/openvdn/health` 一致，避免请求临时重载模型。旧 `warmup_steps` 字段保留兼容，常驻服务统一在启动执行 8 NFE，请求中不再额外预热。

固定 video/audio shift=12/3、8 NFE，不提供无对应权重的步数切换。`reference_short_edge=2048` 会显著增加参考 token 和显存。

## 环境、日志与验证

两套环境：`.venv-ui` 为 ComfyUI 0.30.0、CPU torch 2.10；`.venv-vdn` 为官方 torch 2.13.0+cu129、Transformers 5.15、FlashAttention 4。**UI 日志的 `Device: cpu` 是预期行为，CUDA 由常驻 worker 使用。** 精确源码/模型 revision 见 `sources.lock.json`，Diffusers 使用官方指定 base 和补丁；生成时离线读取固定权重。

- 后端加载/推理日志：`.runtime/backend/worker.log`
- GPU 清理记录：`.runtime/gpu-cleanup.json`
- API 状态：`.runtime/api/jobs/`
- 视频：`output/openvdn/*.mp4`；官方计时/内核状态：`视频.mp4.inference.json`
- 请求耗时、缓存命中、实际规格：`视频.mp4.metrics.json`
- 请求配置/结果：`.runtime/jobs/<job_id>/`
- 条件及编译缓存：`.runtime/conditioning/`、`.runtime/inductor/`、`.runtime/triton/`
- UI 数据库：`.runtime/comfy-user/comfyui.db`，启动显式指定并创建父目录。

worker 异常时 UI/API 会返回出错 rank 的独立堆栈，并附日志末尾（最多 32 KiB / 160 行）。对比去噪速度看 `metrics.timings.denoise_seconds`；REST 从入队到完成看 `metrics.timings.api_wall_seconds`，实际处理看 `processing_wall_seconds`（排除 ComfyUI 排队）。旧 `request_wall_seconds` 保留，排除参考图下载和 ComfyUI 排队。官方报告的 H200 18.3 秒是去噪耗时，并不是本项目实测端到端耗时。[官方结果](https://github.com/OpenVDN/vdn-minimax-h3#results)

本地测试覆盖请求参数、尺寸/音频裁剪、URL 校验、REST 队列与状态、常驻进程 mailbox/取消、PID 身份、GPU 服务清理分支，以及官方 Ulysses 连续切换序列长度。CPU 测试无法验证 CUDA 内核、峰值显存、Qwen 多卡分配和最终画质；这些需在 8×H200 实测。已有两次服务器尝试均受其他 SGLang 服务占显存影响，没有成功视频，也没有本部署的速度结论。


## 输出优化与阶段耗时（schema 2）

使用原版视频/音频 VAE。输出阶段在 GPU 按 8 帧处理颜色、缩放、uint8 转换，提前裁掉超出目标时长的帧；只复制目标尺寸的 RGB 到 CPU。一个有界预取线程将下一批像素准备与当前批 CPU H.264 编码重叠，避免原流程整段 float32 像素展开和 CPU 插值。MP4 仍原子提交，失败不留下可被误认成功的文件。

`GET /openvdn/jobs/{id}` 成功结果的 `metrics.timings` 返回秒数：

| 字段 | 含义 |
| --- | --- |
| `api_queue_seconds` / `gpu_queue_seconds` | ComfyUI 入队等待 / 共享 GPU 锁等待 |
| `reference_download_seconds` | 图片下载、校验和本地缓存 |
| `conditioning_seconds` | 文本/图像条件编码与条件缓存；命中时接近零 |
| `condition_load_seconds` | 条件张量加载至 GPU 和形状准备 |
| `denoise_seconds` / `step_seconds` | 8 步采样总耗时 / 每一步 GPU 同步耗时 |
| `video_vae_decode_seconds` / `audio_vae_decode_seconds` | GPU 视频 / 音频 VAE 解码 |
| `pixel_prepare_seconds` | GPU 颜色转换、缩放和 uint8 转换，逐批累计 |
| `device_to_host_seconds` | 音视频 GPU→CPU 传输，逐批累计 |
| `h264_encode_seconds` | RGB→视频帧转换和 CPU H.264 编码，包括 flush |
| `audio_encode_and_mux_seconds` | CPU AAC 编码和音频封装 |
| `mux_seconds` / `output_commit_seconds` | 视频封装/容器关闭 / 最终文件原子重命名 |
| `output_wall_seconds` | 完整解码和输出的实际墙钟耗时 |
| `cleanup_seconds` | 张量清理、释放临时 CUDA 缓存和多卡同步 |
| `worker_wall_seconds` / `generation_wall_seconds` | 常驻 worker / 生成调用的墙钟耗时 |
| `processing_wall_seconds` / `api_wall_seconds` | 含下载的处理耗时 / 另含 ComfyUI 排队的总耗时 |

GPU 阶段在计时边界同步。像素准备与 H.264 编码有重叠，**不要把全部组件时间直接相加**；总耗时使用 `*_wall_seconds`。`upstream.output_encoding` 记录实际设备、编码预设和重叠标记，`upstream.new_geometry` 标记首次形状，`conditioning_cache_hit` 标记条件缓存。首次形状与重复形状应分别比较，不能将首次编译耗时误算为模型重载。旧 `decode_and_encode_seconds` 保留（含末尾八卡 barrier）。计时不包含客户端下载生成视频的网络耗时。
