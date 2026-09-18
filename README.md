# OpenVDN 8 步 · ComfyUI · 8×H200

一条视频使用全部 8 张 H200。ComfyUI 负责输入、队列和视频预览；常驻八卡进程调用固定版本的 [OpenVDN](https://github.com/OpenVDN/vdn-minimax-h3)，默认 FP8、6 个 softmax rank + 2 个 linear rank、8 NFE。

图片参考模式是官方 **Ref2VA-like**：FL2VA 权重接收参考图，不是 MiniMax 的独立 Ref2VA transformer。支持默认关闭的 DBCache 跨步缓存、可选 Sol 窗口 softmax；暂不接入 LightX2V 或整块 DiT 编译；`inference_kernels` 控制官方融合/局部编译内核组合。

## 一条命令启动

已部署的服务器更新：

```bash
cd /root/ref2va && git pull --ff-only && REF2VA_TOKEN_BUCKET=2048 bash deploy.sh start
```

首次部署，在克隆的仓库目录执行 `bash deploy.sh`，自动安装固定版本源码、下载模型并启动。需要 Linux x86_64、8 张完整 H200、支持 CUDA 12.9 的驱动、NVLink/NCCL，约 250 GB 磁盘空间。命令在前台运行，可放入 tmux。

当前默认测试 **2048 间隔的无屏蔽前缀分桶**：补齐 token 参与 attention，可能改变生成结果；尚无这版的 H200 耗时/效果实测。上面的更新命令显式覆盖旧环境中的 0/1024 设置；`REF2VA_TOKEN_BUCKET=0 bash deploy.sh start` 可回到不补齐的原生布局。

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
    "reference_short_edge": 768,
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
| `reference_short_edge` | 编码参考图的短边，默认 768，范围 128–2048、32 的倍数；独立于输出 `resolution` |

参考图按该短边等比例放大或缩小，两边四舍五入到 32 的倍数，不裁剪或拉伸到视频比例。例如 `resolution=768, reference_short_edge=512` 是视频短边 768、参考图短边 512。降低参考图短边会减少细节和编码 token，不属于无损优化，不会自动降低。Qwen 图像处理器还会生成自己的视觉网格。

结果 `metrics.upstream.conditioning` 返回参考图实际 `original_size`、`normalized_size`（均为宽、高）、VAE `latent_shape`、`prompt_tokens`、`text_tokens`、`vision_tokens`；新条件缓存另有 `qwen_grid_thw`。旧缓存从 latent 恢复归一化尺寸，原文件已不存在时原尺寸为 null，不重新编码或伪造尺寸。

例如 `duration=10, ratio=9:16, resolution=720` 输出 **720×1280、240 帧、10 秒**。模型内部在 736×1280、243 帧上生成，再缩放到输出尺寸并裁到目标时长，音频同步裁剪。内部宽高对齐 32、帧数对齐 `17n+5`。内部画布面积不超过 1920×1088；支持参数范围不代表所有高分辨率、长时长、多参考组合都能装入显存。

尺寸修复：`bafb944` 至 `b3abe0b` 的 exact runtime 意外固定了默认 1344×768 采样画布，竖屏请求会在输出时被拉伸。更新后采样函数逐请求读取实际画布，并在 VAE 前验证 latent 尺寸；不匹配直接报错。健康接口的 `sampler_geometry=request_bound_v1` 表示修复已加载，结果的 `metrics.upstream.actual_geometry` 返回实际生成尺寸。旧视频仍保留原错误比例；这些版本的非默认尺寸耗时也需重新测试，不能用输出 MP4 尺寸作为采样正确的证明。

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
| `REF2VA_WARMUP_RECENT` | 0 | 默认不回放历史；可手动增加，最大为编译形状容量减 1 再减额外时长数 |
| `REF2VA_WARMUP_DURATIONS` | 空 | 默认不额外预热时长；可设 `5,8,10,15`，最多四个 4–15 秒的值 |
| `REF2VA_WARMUP_VERIFY` | 0 | 默认不复跑预热集验证热命中；1 开启该诊断检查，会增加启动时间 |
| `REF2VA_TOKEN_BUCKET` | 2048 | 无屏蔽前缀分桶间隔，可设 256/512/1024/2048；padding 参与 attention，可能影响效果；0 恢复不补齐的原生布局 |
| `REF2VA_WARMUP_DURATION` | 10 | 启动预热时长 |
| `REF2VA_WARMUP_RATIO` | 9:16 | 启动预热画幅 |
| `REF2VA_WARMUP_RESOLUTION` | 768 | 启动预热短边 |
| `REF2VA_X264_PRESET` | veryfast | CPU H.264 编码预设，可改 medium；保持 CRF 23，更快预设可能增大文件并改变压缩细节 |
| `REF2VA_X264_THREADS` | 8 | H.264 编码线程数，1–64 |
| `REF2VA_REFERENCE_SHORT_EDGE` | 768 | 启动预热参考图短边 |

精度/内核配置需修改环境变量并重启；显式传入 `fp8`、`inference_kernels`、`softmax_backend` 必须与 `/openvdn/health` 一致。`softmax_ranks` 和 `profile` 已改为按请求切换，不重载模型；省略时使用服务启动默认值。旧 `warmup_steps` 字段保留兼容，常驻服务统一在启动执行 8 NFE，请求中不再额外预热。

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


## 编译缓存及启动预热（schema 6）

保留 `.runtime/inductor/`、`.runtime/triton/`，显式开启 PyTorch FX graph/AOTAutograd 磁盘缓存；这些环境变量可由部署覆盖。缓存目录应随部署保留，PyTorch 自己按源码、硬件和编译配置判定能否复用，不在 H200/B200 之间强行复用二进制。

原先每 8 种几何配置就重置全部编译记录，现默认允许 32 种成功配置，并给每个静态 helper 预留多版本编译预算。超过容量才轮换 Dynamo 图，mask 的原有 64 项 LRU 不随之清空。保留超出编译预算直接报错的行为，避免 Flex 静默进入高显存 eager 路径。`geometry_seen` 仅表示进程执行过该输入布局，实际新图编译与磁盘缓存命中另由 PyTorch 计数返回，不能混为一谈。

默认启动完成一次完整 8 NFE、视频/音频 VAE、输出编码和原有数值校验。当前实验允许 padding 参与 attention，因此已移除旧版“补齐位置不影响有效输出”的校验，不再生成 `padding_attention_verified` 或 `token-bucket-parity.json`；旧实例留下的该文件不代表当前版本通过画质验证。VAE、输出传输及常量复用的原有组件校验继续执行。

默认仅完成上述一次基础预热和组件检查就开放服务，不预热全部时长、不回放历史 case、不再复跑整个集合。模型保持 GPU 常驻；进程内图缓存、磁盘编译缓存和条件缓存继续保留。默认将非视频前缀补齐到 2048 的倍数，新计算规格在第一次实际请求时可能编译，后续兼容请求复用；运行时命中率和耗时仍逐次返回。

额外预热可显式开启：`REF2VA_WARMUP_DURATIONS=5,8,10,15` 预热指定时长，`REF2VA_WARMUP_RECENT=27` 回放最近成功请求。历史从本地 `.runtime/backend/warmup-history.json` 和相同源码/模型版本最近 200 个成功 jobs 补齐，直接复用 `.pt`，不重新下载或编码，不输出重复视频。二者默认均关闭。

`REF2VA_WARMUP_VERIFY=1` 可额外复跑所选集合，要求所有 rank 无新图编译才完成启动；默认关闭。健康接口 `startup_warmup.complete` 表示配置要求的启动检查完成；未执行热态复跑时 `verification_requested=false, verified_requests=0, all_runtime_graphs_reused=null`，不会把未检测伪装成全部命中。报告仍写入 `.runtime/backend/warmup-report.json`。

当前实验默认 `REF2VA_TOKEN_BUCKET=2048`，把 `[文本/视觉条件 | 参考图 latent | 音频 | 视频]` 的非视频前缀补到 2048 token 的倍数，最多增加 2047 行。补齐放在音频与生成视频之间，真实 position_ids、参考图几何、文本行、噪声生成顺序均保留。删除 padding `score_mod` 和 attention 包装，恢复上游完整窗口的原生 dispatch 与局部窗口的原生 Flex/FA4 路径；局部窗口 BlockMask 仍限制视频间的注意力范围，但将 gap 视为全局前缀，**不排除补齐 key**。这会改变 softmax 归一化及后续特征，不承诺生成效果与不补齐相同。线性注意力的文本状态仍只读取真实文本行，DBCache 的误差分组也仍排除补齐行；RDT 开关/阈值保持原请求值，但实际缓存决定可能随特征变化。

只对 Flex 配置启用分桶；decomposed/ref 自动使用原始布局。保持 FA4 静态编译，不直接改 `dynamic=True`。旧版带 `score_mod`、1024 桶的同一「走廊功夫」10 秒案例，参考图短边 768、RDT 0.25、两次均无新编译且缓存步骤均为 4/6，分桶前后去噪为 11.80 / 15.08 秒，视频 VAE 均约 2.15 秒；这些不是当前无屏蔽 2048 桶的测量。新策略名为 `prefix_gap_unmasked_v2`，健康接口 `token_bucket_policy` 和每次结果的 `compilation.token_bucket.policy` 可核对部署，后者另返回 `padding_attention=unmasked`。本次没有修改编译形状容量/淘汰策略，仍需部署后复测热态耗时、DBCache 命中和画质。

### 原生路径的缓存命中条件

- 视频时长以对齐后的采样帧数为准，ratio / resolution 以最终推理画布和潜变量形状为准；输入参数不同但最终形状相同不一定重编译。
- 提示词的实际 token 数、文本/视觉 token 排列影响条件长度与布局。相同字数不保证相同 token 数；文字内容本身不作为编译缓存键。
- 参考图短边、长宽比、张数及各图归一化后的 latent / 视觉 token 数影响布局。只替换同处理尺寸的图片像素，不必然产生新编译；条件编码缓存按提示词、图片内容、参考图短边和源码版本另算。
- 当前 `GeometryCache` 管理签名包含 softmax ranks、推理宽高、采样帧数、embeds 形状、完整 tags 序列、参考锚点及各图 latent 形状。它用于管理 32 种成功配置；新签名不等同于底层新编译。容量满后遇到新签名会整体 reset Dynamo，磁盘缓存及 64 项 mask LRU 保留。
- 底层编译还检查 tensor shape / stride / dtype / device、attention 分支和编译配置等 guard。mask LRU 按序列布局、窗口、锚帧、设备及 block 大小区分；mask 命中不等于运行图命中。
- 重启会失去进程内热图；磁盘缓存可能复用，但仍有 tracing / 加载开销。删除缓存目录或改变源码、PyTorch / CUDA / FA4、硬件及编译配置可能使磁盘缓存失效。
- seed、输出文件名通常不改变 attention 编译规格；RDT 阈值主要影响跨步激活复用和实际执行工作量，应与编译缓存命中分开统计。

`metrics.upstream.compilation` 返回：

- `geometry_id` / `geometry_seen` / `successful_geometries`：布局指纹、此前是否成功运行和当前保留数。
- `reset` / `reset_reason` / `generation`：是否因容量轮换及轮换次数。
- `compiled_new_graph`：本次去噪是否有 Dynamo 新图；不是所有底层 JIT 的通用命中标记。
- `runtime_graph_reused`：所有 rank 均无新图且 Dynamo 编译时间增量为零；与磁盘缓存命中分开。`disk_graph_cache_hits/misses` 是各 rank 的计数总和，不是耗时。
- `token_bucket`：真实/补齐后的序列长度、前缀容量、补齐占比、条件（含视觉）/参考 latent/音频/视频 token 数；`geometry_seen` 在分桶模式下表示容量布局相同。
- `by_rank`：各卡的 `unique_graphs`、`fxgraph_cache_hits/misses`、`mask_hits/misses`、`dynamo_compile_seconds`、`mask_build_seconds`。
- `by_rank[].guard_failures/recompile_reasons`：guard 失败计数及本次最近 8 条原因，帮助定位新图由哪些形状/stride/标量变化触发。

`metrics.timings.dynamo_compile_seconds` 和 `mask_build_seconds` 取各卡最大值，不累加并发的八卡时间。前者来自 PyTorch `entire_frame_compile`，包含 tracing/图缓存加载等编译框架工作，不单指 CUDA kernel 编译；后者在 mask 缓存未命中时 GPU 同步计时。两者都嵌套在 `denoise_seconds` 内，彼此也可能重叠，不可加到去噪或总耗时上。

`denoise_wall_seconds` 明确标记去噪阶段墙钟时间（兼容原 `denoise_seconds`）；`hot_denoise_seconds` 仅在本次没有编译时返回实测值，否则为 null，不用减去编译时间的推算值冒充热态实测。

部署后的原模板验收（输入为已有 `case-name/request.json`，不改提示词、图或 RDT 设置）：

```bash
python3 scripts/benchmark_compile_cache.py --server http://43.218.119.131:8188 \
  --requests-dir /path/to/original-cases --output-dir work/cache-benchmark
```

默认顺序跑两遍，允许第一遍首次编译，要求第二遍热命中，输出逐 case 的视频链接、原始响应、`timings.csv` 和 `summary.json`。`--no-allow-first-compile` 可用于严格检查启动后两遍都已热身。报告保持编译/去噪/VAE/端到端耗时分开。

## 输出优化与阶段耗时

使用原版视频/音频 VAE。输出阶段在 GPU 按 8 帧处理颜色、缩放、uint8 转换，提前裁掉超出目标时长的帧；只复制目标尺寸的 RGB 到 CPU。默认使用两个固定大小的 pinned CPU 缓冲区，独立 CUDA stream 执行像素准备和非阻塞 D2H，CPU 同时编码已完成的批次。消费者只等待该批次的完成事件，编码结束后才复用缓冲区；不在每批前后同步整个 GPU。保留像素运算顺序、舍入、插值、libx264 `veryfast` / CRF 23 / 8 线程和 AAC 参数。MP4 仍原子提交，失败不留下可被误认成功的文件。

`REF2VA_ASYNC_OUTPUT=0` 恢复原有单预取线程输出；CPU 测试也使用此路径。启动合成案例逐批比较异步传回的 RGB 与原同步路径，要求逐元素一致；失败则不开放 UI。`upstream.output_encoding` 返回 `async_pinned_output`、`pixel_timing_method` 和 `pixel_parity`。GPU 对照测试还覆盖非默认生产流、缓冲区复用、非整批尾帧，以及同步/异步输出 MP4 的解码后音视频一致性；在没有 CUDA 的环境中明确跳过，不计为通过。

## 单次请求内的 DiT 去重与同步优化（schema 4）

默认 `REF2VA_EXACT_RUNTIME=1`。默认关闭 DBCache 时完整运行 8 次 DiT，保留所有注意力和线性分支、FP8 设置、权重及采样器，不启用 Sol 或跨步残差缓存：

- 同一次请求的 `RoPE(position_ids)`、`token_refiner(context_embedder(prompt_embeds))` 只计算一次，其余 7 次复用。输入存储、形状、stride、版本、dtype、设备与 autocast 变化会失效；请求结束或异常立即释放。不同请求不共享这些结果。
- 每层线性输出投影的 GPU 布尔索引/`any()`/`sum().item()` 改为 CPU 已知区间切片，保留原 GEMM 的输入形状、连续布局和精度。
- 每步计时使用 CUDA event，循环结束统一读取，去掉仅为计时增加的逐步全设备同步。模型本身需要的同步不变；没有改变完整隐藏特征的 gather 和输出投影次序。

适配层只接受固定 OpenVDN 函数的完整源码 hash，保持 `.deps` 工作区不变；源码或注意力方法不匹配直接拒绝启动。启动及历史预热时，在相同输入上对比缓存常量与重算结果、每个 attention 模块第一次输出投影与原始布尔索引路径，要求有限且逐元素一致。各卡先完成去噪并交换校验状态，再统一报错，避免某一卡在 collective 前退出。记录位于 `.runtime/backend/exact-runtime-parity.json`、`warmup-report.json` 以及每次结果的 `upstream.exact_runtime.by_rank`。这是组件和预热案例校验，不是所有提示词的端到端质量证明；本地 CPU 通过也不代表 H200 已测性能。

两个开关都只在启动配置：需要回退本轮优化时执行 `REF2VA_EXACT_RUNTIME=0 REF2VA_ASYNC_OUTPUT=0 bash deploy.sh start`，其余模型、并行 VAE 和编译缓存配置不变。`/openvdn/health` 返回开关及 `metrics_schema_version: 5`。

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

## 按请求分析多卡耗时与 DBCache（schema 5）

更新仍只需一条命令，无新增依赖：

```bash
cd /root/ref2va && git pull --ff-only && bash deploy.sh start
```

### 多卡布局与分析

`softmax_ranks` 现在可逐请求指定：6 为 6+2，5 为 5+3，4 为 4+4，0 为普通八卡 Ulysses。只在串行队列的请求边界切换分工，保留权重与 communicator。不同布局单独记录编译几何；首次切换可能编译，随后复用。服务默认仍为 6+2，比较脚本不会替用户永久更改默认值。改变并行分工不引入缓存近似，但浮点运算顺序可能不同，不能承诺逐像素一致。

`profile: true` 按请求开启 CUDA event 分析，默认关闭。返回 `metrics.upstream.parallel_profile`：

- `by_rank`：每卡角色、head 数、各段 `total_ms`、`ms_per_nfe`、调用次数。
- 分段包含输入准备、所有 DiT blocks、attention、FFN、末尾 gather、输出 head；分支路径进一步包含 QKV、gate、打包、分发等待、softmax/linear 计算、回传及输出投影。
- `branches` / `max_ms_per_nfe` 用于找慢卡和分支不均衡。计时测到的是计算流上的时间跨度，包含可见等待和提交间隙，并不是独立 NCCL kernel 的纯耗时。
- `branch_dispatch` 包含 `branch_pack` 和 `branch_relevant_wait`，`output_dispatch` 包含 `output_a2a` 和 `output_unpack`，`blocks` 包含 attention/FFN；这些层级有重叠，不能相加，也不能累加八卡计时作为请求耗时。分析会有额外开销，正式测速使用 `profile: false`。

### Cache-DiT / DBCache 参数

这是按 [Cache-DiT DBCache 算法](https://github.com/vipshop/cache-dit/tree/main/src/cache_dit/caching/cache_blocks) 独立实现的 OpenVDN 八卡适配层 `openvdn_dbcache_adapter_v1`，不是直接安装其 Python 包或 ComfyUI 原生 H3 插件，不启用 TaylorSeer。先完整计算前 Fn 层，比较前缀残差与上次完整计算时的前缀残差；足够相似时复用中间层残差，再完整计算最后 Bn 层。保留原模型的混合 attention、参考图条件、音频、8 次采样调用和后处理。缓存命中会改变去噪轨迹，效果需逐案例对照。

所有参数都可通过 REST / ComfyUI / CLI 按请求设置，不用重启：

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `cache_dit` | false | 开关；关闭时无残差缓存拷贝或额外决策 collective |
| `cache_dit_threshold` | 0.08 | 复用变化阈值，0–1；越小越保守，0 完全不复用；1 仍需通过变化量检查 |
| `cache_dit_fn_blocks` | 8 | 每步完整计算前 Fn 层，1–49 |
| `cache_dit_bn_blocks` | 8 | 每步完整计算后 Bn 层，0–49；Fn+Bn 必须小于 50 |
| `cache_dit_warmup_steps` | 3 | 前几次 DiT 调用完整计算，1–8；不是额外增加采样步数 |
| `cache_dit_max_consecutive` | 1 | 最多连续复用几步，1–7 |
| `cache_dit_max_cached_steps` | 2 | 本次请求总复用步数上限，0–7；0 不复用 |
| `cache_dit_last_steps` | 1 | 最后几步强制完整计算，0–7；与 warmup 之和不超过 8 |

默认参数是 8 步模型的保守测试起点，尚未在 H200 上验证收益/效果，不是已验证的质量预设。不同于直接把所有 token 混合求均值，适配层分别统计 text、reference、目标 video、audio 的相对 L1 变化，取最大值判断；防止长视频 token 掩盖音频或参考条件的变化。每张卡先计算局部分子/分母，再全局 SUM，兼容不等长序列分片。所有卡得到相同判断后才能跳过中间层的 collective；任一卡缓存缺失、残差非有限或不兼容都会全体回到完整计算。每次请求结束或失败立即清空缓存；启动及历史预热强制关闭 DBCache，完整预热全部 8 步并执行原有数值检查。

向原请求加入如下字段即可开启：

```json
{
  "softmax_ranks": 6,
  "profile": false,
  "cache_dit": true,
  "cache_dit_threshold": 0.08,
  "cache_dit_fn_blocks": 8,
  "cache_dit_bn_blocks": 8,
  "cache_dit_warmup_steps": 3,
  "cache_dit_max_consecutive": 1,
  "cache_dit_max_cached_steps": 2,
  "cache_dit_last_steps": 1
}
```

返回 `metrics.upstream.cache_dit`：实际参数、`cache_hits`、`cached_steps`（1-based）、`full_steps`、执行/跳过的 block 数、每步原因、各模态误差代理值及 `all_rank_agreement`。`approximate: true` 表示这次实际发生了复用；开启但零命中时不会声称加速。L1 阈值是判断代理量，不是输出画质误差界。

### 自动对比脚本

保存一份正常的 POST 请求为 `case.json`，可在本地或服务器执行：

```bash
python3 scripts/benchmark_optimizations.py \
  --server http://43.218.119.131:8188 --request-file case.json \
  --output-dir work/optimization-benchmark \
  --layouts 6 5 4 --thresholds 0.04 0.08 0.12 --repeat 3
```

依次比较 6+2、5+3、4+4 的无缓存热态速度，每种 1 次预热、3 次测速、1 次独立 profile；再在实测最快且热态没有新图编译的布局上比较三个缓存阈值，每种 1 次预热、3 次测速。整轮默认生成 27 条视频，串行使用同一服务；不重启、不结束其他应用。所有参数保持原请求，除布局/分析开关/缓存开关和阈值外不修改提示词、参考图、seed、尺寸或时长。启动实例变化或任务失败立即停止，不静默重试生成。已有结果目录拒绝覆盖。

输出 `results.json`、`summary.json`、`report.md` 和每次请求/完整响应，包括视频链接、缓存命中数、编译情况、每卡分析。报告的耗时只统计独立热态样本中位数，不混入预热或 profile；须另外观看缓存与无缓存视频比较动作、身份和音画同步。未完成 H200 实测前，不承诺某种布局或缓存阈值一定更快。

## Attention、通信与流式输出优化（schema 7）

```bash
cd /root/ref2va && git pull --ff-only && bash deploy.sh start
```

没有新增依赖。启动仍只做一个基础样例，不全量预热各种内核/形状。通信优化启用前，在每张卡上用小张量对比原 pack/unpack，并实际执行一次不等长 NCCL 往返；任意卡不一致就停止启动。视频输出继续执行原有真实 VAE clip 校验，并额外核对流式 RGB 像素与完整视频后处理的结果。首次运行新的内核/参数会编译，之后复用。

以下字段可通过 REST、ComfyUI 或 CLI 逐请求切换：

| 字段 | 默认 | 作用 |
|---|---|---|
| `fast_communication` | `true` | 每层从逐目标打包改成两个 Triton kernel，并融合回传后的 head 重排。NCCL payload、顺序、精度和异步 work/buffer 生存期不变；不量化通信。只作用于 branch-parallel 布局。 |
| `attention_kernel` | `native` | `native` 保留当前 FA4/Flex 路径；`decomposed` 用上游 dense + varlen 分解相同窗口，保留相同桶。需要比较速度和浮点误差，不自动宣称更快。 |
| `linear_stats_chunk_frames` | `16` | 可选 `8/16/32`，调整 linear 分支逐帧统计的 GEMM 批次，方程与精度不变。32 减少批次数、增加峰值显存；cuBLAS 算法可能随批次改变。 |
| `isolate_padding` | `false` | 仅在 `softmax_backend=flex`、`attention_kernel=native` 时可用。用 BlockMask 排除中间 gap 的 key，保持完整块的快速路径，不使用 score_mod；full-cover 情况也通过隔离 mask，不激活额外 linear 分支。是否满足低开销目标须实测。 |
| `streaming_output` | `true` | 并行 VAE 边解码边传回 clip；rank 0 按原生时间混合顺序提交有效帧，CPU H.264 与后续解码重叠。编码队列最多 4 个像素块，非零卡最多保留 2 个异步发送；异常会退出线程并删除 partial MP4。关闭时恢复整段解码后编码。 |
| `cleanup_policy` | `adaptive` | 热态请求结束保留 CUDA 空闲内存池。新编译、启动校验、显存压力或每 32 次请求清理；`always` 恢复每次 gc/empty_cache。新的条件编码前所有卡仍释放空闲缓存，给 Qwen 跨卡临时分配腾出空间。 |

服务启动默认值也可通过 `REF2VA_FAST_COMMUNICATION`、`REF2VA_ATTENTION_KERNEL`、`REF2VA_LINEAR_STATS_CHUNK_FRAMES`、`REF2VA_ISOLATE_PADDING`、`REF2VA_STREAMING_OUTPUT`、`REF2VA_CLEANUP_POLICY` 设置。若要完全恢复此前调度，用 `REF2VA_FAST_COMMUNICATION=0 REF2VA_STREAMING_OUTPUT=0 REF2VA_CLEANUP_POLICY=always bash deploy.sh start`。现有 `REF2VA_ASYNC_OUTPUT` 控制非流式输出的 pinned-memory 预取。

返回 `metrics.upstream.optimizations`，包含实际参数、通信小张量/真实 NCCL 一致性检查、attention 选择和清理原因。隔离开启时 `compilation.token_bucket.policy=prefix_gap_isolated_v3`，`padding_attention=excluded_keys`；默认仍为 `prefix_gap_unmasked_v2`。隔离与原生不补齐在有效 key 集合上等价，浮点核/矩阵尺寸不同，不能承诺生成视频逐像素一致。Cache-DiT 仍按原参数独立运行。

流式模式的 `video_vae_decode_seconds` 是包含传输、拼接、像素提交/队列反压的阶段 wall time，不是纯 VAE GPU 算子时间。新增 `video_vae_compute_max_rank_seconds` 来自逐 clip CUDA event，取八卡中最大的累计 decode 时间；`video_vae_decode.by_rank` 保留各卡明细。`output_pipeline_wall_seconds` 是整个重叠流水线耗时，`first_pixel_chunk_seconds` 表示第一批像素交给编码器的延迟。H.264、VAE、D2H 等组件现在有更多重叠，**不累加这些字段估算总耗时**；使用 `processing_wall_seconds` / `decode_and_encode_seconds` 比较端到端结果。

### 可恢复的逐项对照

用同一个 768 参考图、768 输出、RDT 0.25 的请求文件：

```bash
python3 scripts/benchmark_attention_pipeline.py --server http://43.218.119.131:8188 \
  --request-file case.json --output-dir work/attention-pipeline --repeat 3
```

默认 8 组（原调度、仅通信、仅流式、仅清理、三者合并、合并+decomposed、合并+linear32、合并+隔离），每组 1 次冷启动/预热和 3 次交错热态样本，总共 32 条。可用 `--variants baseline combined` 先跑 8 条。请求内容、seed、时长、分辨率、参考图和 Cache-DiT 参数保持一致。热态仍触发 Dynamo 编译的组标为无效，不参与速度结论。输出请求、完整状态、视频链接、分步耗时、缓存实际跳步、`summary.json` 和 `report.md`。同目录可恢复已接受的任务，不会因轮询中断重新提交生成；接受状态不确定的 POST 会停止并要求检查队列。

可选、不加载模型的八卡 kernel 校验（请在 GPU 空闲时运行，不属于默认启动全量预热）：

```bash
.venv-vdn/bin/torchrun --standalone --nproc_per_node=8 scripts/validate_optimization_kernels.py
```

覆盖 1+7 至 7+1 的 pack/unpack/NCCL，以及 FA4 隔离窗口/full-cover 和 decomposed 对 fp32 dense 参考的容差校验。**本地 CPU 回归通过不等于 H200 新路径已通过；只有服务器校验与热态对照完成后才能报告实际加速。**

## Sol 窗口 attention（schema 8，H100/H200 实验后端）

更新并补齐可选依赖，只执行这一条；不重新下载模型，不更换 CUDA PyTorch：

```bash
cd /root/ref2va && git pull --ff-only && bash deploy.sh install-sol && bash deploy.sh start
```

默认仍使用 `attention_kernel=native`，启动不会全量预热 Sol。首次请求选择 Sol 时，各 rank 先执行独立数学参考校验，包括 Q/K 不等长、尾块、全精确及稀疏分支；失败明确报错，不静默退回原 attention。第一次出现的形状可能编译，之后复用进程内 CuTe callable 和 Triton 编译缓存。CuTe callable 最多保留 128 个形状（LRU）；不把重启后首次加载宣称为进程内命中。依赖导入通过不等于 H200 数值/性能验证通过。

`install-sol` 使用 FA4 `4.0.0b26` / quack 要求的 `nvidia-cutlass-dsl==4.6.0.dev0`，显式允许该预发行版。它保留已安装的 torch（包括 cu129 后缀）、torchvision、triton、FA4、quack 版本，先联合解析依赖，再安装并检查三套内核的导入。如果此前被 Sol 安装命令升到 4.7.1，再执行同一条更新命令即可：脚本会先移除 4.7 拆出的 `libs-core` / `libs-cu12`，再恢复 4.6，避免共享路径残留；不通过跳过依赖检查掩盖冲突。

在原请求中增加：

```json
{
  "attention_kernel": "sol",
  "sol_tau": 1.0,
  "sol_dense_steps": 1,
  "sol_dense_layers": 2,
  "softmax_ranks": 6
}
```

- `attention_kernel`：`native / decomposed / sol`，支持 REST、ComfyUI 和 CLI 逐请求切换。
- `sol_tau`：0–4，默认 1.0。阈值越大，通常越多局部 K/V 块使用质心近似。0 也不是全精确模式；要关闭 Sol 用 `native`。这是 Sol 路由阈值，和 `cache_dit_threshold` 的 RDT 无关。
- `sol_dense_steps`：前多少次 DiT 调用沿用原 attention，0–8，默认 1。
- `sol_dense_layers`：每次 DiT 调用的前多少层沿用原 attention，0–50，默认 2。两项任一达到最大值就不会实际执行 Sol。
- `isolate_padding` 可与 Sol 同开。原路径的保留层/步用现有 BlockMask，Sol 窗口先排除 padding key 再计算；不通过 score_mod，也不将 Q 补成 K 的长度。排除 padding 会改变物理 K 长度，因此原始 prefix 变化可能产生新的 Sol 形状。

实现固定 NVIDIA [Sol-H3 SM90 源码](https://github.com/NVlabs/Sana/tree/bb60499af0e675095ff67424196d8c18e265f32a/models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn)，在其矩形 host 适配层按相同 Q/K 长度批量处理 VDN 窗口。**只近似局部视频 softmax 的一部分 key 块**；保持原窗口可见域，文本、参考、音频及 anchor keys 作为精确 sink（向外对齐 64，最多额外保留 63 个局部 key），全局/anchor query 行精确计算。VDN 线性分支、gate、输出投影、RoPE、权重及 8 NFE 不变。原先 full-cover 层仍用原生 dense 分支。Sol 是近似优化，不能保证画质或数值等同；默认参数是测试起点，不是已验证质量预设。

可继续使用 Cache-DiT RDT 0.25。切换 native→Sol 的那一步会在所有 rank 清空旧残差，避免复用上个计算阶段的残差；其后仍按 RDT 实际判断。首次效果对照建议先保持相同 seed/提示词/参考图和缓存参数，同时记录实际缓存步数；要单独分析 Sol 的误差，再关闭 Cache-DiT 比较。

返回 `metrics.upstream.optimizations.attention`：

- `sol.sparse_executed`、`sparse_kernel_launches_all_ranks`：是否真的调用，以及各 rank 调用总数。调用不等于加速或测得稀疏率；不在热路径额外同步统计每个选中块。
- `by_rank[].sol`：原 attention 的步/层保留原因、dense/window query 行数、保护 key 数、过渡残差清理、校验结果、源码版本和实际参数。
- `compilation.sol_compile_misses / sol_preprocess_signatures`：CuTe 首次形状、预处理首次签名。任一非零就把 `runtime_graph_reused` 标为 false，不能混进热态结论。
- `timings.sol_compile_seconds` 是各 rank 最大 CuTe 编译/加载墙钟时间；`sol_preprocess_cold_seconds` 包括首次预处理的编译、autotune 和执行，不是纯编译时间。两项已包含在去噪耗时里，不可重复相加。首次小张量校验在 `condition_load_seconds` 内，另记 `verification_seconds`。

可选：空闲时运行不加载模型的八卡校验，覆盖 6+2、5+3、4+4 的不等长 head 分片尺寸：

```bash
.venv-vdn/bin/torchrun --standalone --nproc_per_node=8 scripts/validate_sol_attention.py
```

部署后用同一份请求跑热态对照（默认 native/Sol 两组，各 1 次冷、3 次热；不重启）：

```bash
python3 scripts/benchmark_sol_attention.py --server http://43.218.119.131:8188 \
  --request-file case.json --output-dir work/sol-benchmark --repeat 3
```

可加 `--layouts 6 5 4 --taus 0.5 1 1.5` 检查软最大值分支加速后的负载平衡。脚本保留输入和 Cache-DiT 参数，保存完整请求/响应、视频链接、分步耗时及缓存步数；拒绝把没有实际 Sol 调用、热态仍编译或跨服务实例的结果当成有效对照。`results.json` 包含冷态开销，`summary.json` 只统计有效热态中位数。本地 CPU 验证不包含 CUDA 内核执行，实际提速需在 H200 实测。
