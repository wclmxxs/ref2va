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
4. 加载八份 DiT、默认八份视频 VAE 和 rank 0 的音频 VAE；Qwen3-VL 条件编码器通过 Accelerate 分配到这 8 张 GPU，单卡权重预算 12 GiB，禁止 CPU/磁盘权重卸载。
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
| `REF2VA_VAE_PARALLEL` | 1 | 八卡视频 VAE 片段并行及预分配 tile 拼接；启动时逐片段对照未优化原版，校验通过才开放服务；0 恢复原版单卡 |
| `REF2VA_COMPILE_SHAPES` | 32 | 编译图轮换前保留的成功几何配置数，8–64；达到容量后才重置 Dynamo，保留磁盘缓存和 mask LRU |
| `REF2VA_WARMUP_RECENT` | 8 | 启动时额外预热最近成功形状，0 关闭，最大为编译形状容量减 1；增加启动时间以减少首次业务请求耗时 |
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

本地测试覆盖请求参数、尺寸/音频裁剪、URL 校验、REST 队列与状态、常驻进程 mailbox/取消、PID 身份、GPU 服务清理分支，以及官方 Ulysses 连续切换序列长度。CPU 测试无法验证 CUDA 内核、峰值显存、Qwen 多卡分配和最终画质；这些需在 8×H200 实测。2026-09-17 已完成 13 个原模板：短边 768、8–15 秒，排除排队的处理耗时中位数 46.31 秒；包含不同输入首次编译，不能作为纯热态性能。

## 默认八卡视频 VAE 解码

```bash
cd /root/ref2va && git pull --ff-only && bash deploy.sh start
```

单卡原版 VAE 将 10 秒视频（内部 243 帧）拆成 14 个独立的 `_decode_clip` 调用。默认将这些调用轮流分配给八卡，每卡常驻同一份原版 float32 VAE 权重，继续使用 float16 autocast。rank 0 接收解码结果后调用原版 `decode()` 完成补帧、重叠融合和尾部裁剪；音频和视频编码保持原流程。不改变采样步数、参考图条件、模型权重或空间分块方式。

每个时间片段内部仍使用原版空间 tile 和重叠区域。借鉴 SGLang 路径，空间拼接先分配最终张量，再逐块拷入，省掉各行 `cat` 再整幅 `cat` 的大张量拷贝；小型融合权重按设备、精度、重叠长度缓存。先纵向再横向融合、浮点乘加顺序均保持原版。这里只并行完整时间片段，不对 ViT 层做空间切分，也不新增 padding 或改注意力后端。

启动预热额外逐片段执行 rank 0 未优化的原版解码，要求所有并行/优化片段与原版逐元素一致且有限；不一致时启动失败，UI 不会显示就绪。校验记录保存在 `.runtime/backend/vae-parity.json`。失败可用 `REF2VA_VAE_PARALLEL=0 bash deploy.sh start` 恢复单卡解码。CPU 测试通过不代表已完成 H200 性能验证，实测前不承诺加速比。

`/openvdn/health` 的 `video_vae_world_size` 返回 1 或 8。每次生成的 `metrics.upstream.video_vae_decode` 返回实际卡数、片段数、`spatial_tiles_by_rank`、各卡计算时间和传输/组装时间；`video_vae_decode_seconds` 仍为完整视频解码墙钟时间，`output_wall_seconds` 包含它且只计一次。启动校验时间包含原版对照，不应用来判断生成速度。

在版本 61245b4 上，同一 10 秒原模板热态处理 29.74 秒，其中采样 13.78 秒、视频 VAE 12.99 秒。提示词新增一句后首次处理 49.83 秒，再次处理 30.14 秒；前两步的额外开销与静态编译/掩码构建相关，尚未单独计量编译时间。下一步应分别比较八卡 VAE 和编译复用，避免把条件缓存命中或首次编译差异算成解码加速。


## 编译缓存及启动预热（schema 3）

保留 `.runtime/inductor/`、`.runtime/triton/`，显式开启 PyTorch FX graph/AOTAutograd 磁盘缓存；这些环境变量可由部署覆盖。缓存目录应随部署保留，PyTorch 自己按源码、硬件和编译配置判定能否复用，不在 H200/B200 之间强行复用二进制。

原先每 8 种几何配置就重置全部编译记录，现默认允许 32 种成功配置，并给每个静态 helper 预留多版本编译预算。超过容量才轮换 Dynamo 图，mask 的原有 64 项 LRU 不随之清空。保留超出编译预算直接报错的行为，避免 Flex 静默进入高显存 eager 路径。`geometry_seen` 仅表示进程执行过该输入布局，实际新图编译与磁盘缓存命中另由 PyTorch 计数返回，不能混为一谈。

启动先完成一次完整 8 NFE、视频/音频 VAE、输出编码和数值校验，再用 `.runtime/backend/warmup-history.json` 中最近 8 个成功配置预热完整 8 NFE。历史预热直接加载已有条件 `.pt`，不重新下载图片或编码，也不导出重复视频。首次升级会从相同源码/模型版本的 `.runtime/jobs/*/result.json` 成功记录迁移；缺失缓存、失败记录和不匹配配置会跳过。固定启动参考图的条件缓存也按内容/源码版本复用。所有预热完成后才开放服务，记录在 `.runtime/backend/warmup-report.json`。

本版不做 4096 token padding，也不把 FLASH Flex 改为动态形状：当前 OpenVDN 的有效 conditioning 和 mask 布局不能直接套用 SGLang 的分桶规则。新布局仍可能首次编译；优化的是已有布局的复用、重启预热和可观测性。

`metrics.upstream.compilation` 返回：

- `geometry_id` / `geometry_seen` / `successful_geometries`：布局指纹、此前是否成功运行和当前保留数。
- `reset` / `reset_reason` / `generation`：是否因容量轮换及轮换次数。
- `compiled_new_graph`：本次去噪是否有 Dynamo 新图；不是所有底层 JIT 的通用命中标记。
- `by_rank`：各卡的 `unique_graphs`、`fxgraph_cache_hits/misses`、`mask_hits/misses`、`dynamo_compile_seconds`、`mask_build_seconds`。

`metrics.timings.dynamo_compile_seconds` 和 `mask_build_seconds` 取各卡最大值，不累加并发的八卡时间。前者来自 PyTorch `entire_frame_compile`，包含 tracing/图缓存加载等编译框架工作，不单指 CUDA kernel 编译；后者在 mask 缓存未命中时 GPU 同步计时。两者都嵌套在 `denoise_seconds` 内，彼此也可能重叠，不可加到去噪或总耗时上。

## 输出优化与阶段耗时

使用原版视频/音频 VAE。输出阶段在 GPU 按 8 帧处理颜色、缩放、uint8 转换，提前裁掉超出目标时长的帧；只复制目标尺寸的 RGB 到 CPU。默认使用两个固定大小的 pinned CPU 缓冲区，独立 CUDA stream 执行像素准备和非阻塞 D2H，CPU 同时编码已完成的批次。消费者只等待该批次的完成事件，编码结束后才复用缓冲区；不在每批前后同步整个 GPU。保留像素运算顺序、舍入、插值、libx264 `veryfast` / CRF 23 / 8 线程和 AAC 参数。MP4 仍原子提交，失败不留下可被误认成功的文件。

`REF2VA_ASYNC_OUTPUT=0` 恢复原有单预取线程输出；CPU 测试也使用此路径。启动合成案例逐批比较异步传回的 RGB 与原同步路径，要求逐元素一致；失败则不开放 UI。`upstream.output_encoding` 返回 `async_pinned_output`、`pixel_timing_method` 和 `pixel_parity`。GPU 对照测试还覆盖非默认生产流、缓冲区复用、非整批尾帧，以及同步/异步输出 MP4 的解码后音视频一致性；在没有 CUDA 的环境中明确跳过，不计为通过。

## 单次请求内的 DiT 去重与同步优化（schema 4）

默认 `REF2VA_EXACT_RUNTIME=1`。本版完整运行 8 次 DiT，保留所有注意力和线性分支、FP8 设置、权重及采样器，不启用 Sol 或跨步残差缓存：

- 同一次请求的 `RoPE(position_ids)`、`token_refiner(context_embedder(prompt_embeds))` 只计算一次，其余 7 次复用。输入存储、形状、stride、版本、dtype、设备与 autocast 变化会失效；请求结束或异常立即释放。不同请求不共享这些结果。
- 每层线性输出投影的 GPU 布尔索引/`any()`/`sum().item()` 改为 CPU 已知区间切片，保留原 GEMM 的输入形状、连续布局和精度。
- 每步计时使用 CUDA event，循环结束统一读取，去掉仅为计时增加的逐步全设备同步。模型本身需要的同步不变；没有改变完整隐藏特征的 gather 和输出投影次序。

适配层只接受固定 OpenVDN 函数的完整源码 hash，保持 `.deps` 工作区不变；源码或注意力方法不匹配直接拒绝启动。启动及历史预热时，在相同输入上对比缓存常量与重算结果、每个 attention 模块第一次输出投影与原始布尔索引路径，要求有限且逐元素一致。各卡先完成去噪并交换校验状态，再统一报错，避免某一卡在 collective 前退出。记录位于 `.runtime/backend/exact-runtime-parity.json`、`warmup-report.json` 以及每次结果的 `upstream.exact_runtime.by_rank`。这是组件和预热案例校验，不是所有提示词的端到端质量证明；本地 CPU 通过也不代表 H200 已测性能。

两个开关都只在启动配置：需要回退本轮优化时执行 `REF2VA_EXACT_RUNTIME=0 REF2VA_ASYNC_OUTPUT=0 bash deploy.sh start`，其余模型、并行 VAE 和编译缓存配置不变。`/openvdn/health` 返回开关及 `metrics_schema_version: 4`。

`GET /openvdn/jobs/{id}` 成功结果的 `metrics.timings` 返回秒数：

| 字段 | 含义 |
| --- | --- |
| `api_queue_seconds` / `gpu_queue_seconds` | ComfyUI 入队等待 / 共享 GPU 锁等待 |
| `reference_download_seconds` | 图片下载、校验和本地缓存 |
| `conditioning_seconds` | 文本/图像条件编码与条件缓存；命中时接近零 |
| `condition_load_seconds` | 条件张量加载至 GPU 和形状准备 |
| `denoise_seconds` / `step_seconds` | 8 步采样同步墙钟耗时 / 每步 CUDA event 时间，见 `step_timing_method`；关闭新路径时为原同步墙钟时间 |
| `dynamo_compile_seconds` / `mask_build_seconds` | 去噪期间 Dynamo 编译框架时间 / GPU 同步的 mask 构建时间；已包含于去噪，可能互相重叠 |
| `video_vae_decode_seconds` / `audio_vae_decode_seconds` | GPU 视频 / 音频 VAE 解码 |
| `pixel_prepare_seconds` | GPU 颜色转换、缩放和 uint8 转换，逐批累计 |
| `device_to_host_seconds` | 音视频 GPU→CPU 传输，逐批累计 |
| `pixel_prefetch_wait_seconds` | 异步输出中 CPU 等待当前 RGB 批次完成的累计墙钟时间 |
| `h264_encode_seconds` | RGB→视频帧转换和 CPU H.264 编码，包括 flush |
| `audio_encode_and_mux_seconds` | CPU AAC 编码和音频封装 |
| `mux_seconds` / `output_commit_seconds` | 视频封装/容器关闭 / 最终文件原子重命名 |
| `output_wall_seconds` | 完整解码和输出的实际墙钟耗时 |
| `cleanup_seconds` | 张量清理、释放临时 CUDA 缓存和多卡同步 |
| `worker_wall_seconds` / `generation_wall_seconds` | 常驻 worker / 生成调用的墙钟耗时 |
| `processing_wall_seconds` / `api_wall_seconds` | 含下载的处理耗时 / 另含 ComfyUI 排队的总耗时 |

去噪总计、VAE 等阶段仍在边界同步；每步、异步像素处理和视频 D2H 改为 CUDA event。event 时间可包含等待和主机未及时提交造成的间隙，不能当作纯 kernel 耗时之和；音频 D2H 仍采用原同步计时。像素准备、传输与 H.264 编码有重叠，**不要把全部组件时间直接相加**；总耗时使用 `*_wall_seconds`。`upstream.output_encoding` 记录实际设备、编码预设和重叠标记，`upstream.new_geometry` 标记首次形状，`conditioning_cache_hit` 标记条件缓存。首次形状与重复形状应分别比较，不能将首次编译耗时误算为模型重载。旧 `decode_and_encode_seconds` 保留（含末尾八卡 barrier）。计时不包含客户端下载生成视频的网络耗时。
