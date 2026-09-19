# OpenVDN 8 步 · ComfyUI · H200 / B200 / B300

默认自动识别 H200/B200/B300，一条视频使用全部 8 张 GPU；可切换为两套四卡服务。ComfyUI 负责输入、队列和视频预览；常驻 GPU 进程调用固定版本的 [OpenVDN](https://github.com/OpenVDN/vdn-minimax-h3)，默认 FP8、8 NFE；默认 `softmax_ranks=0`，使用双流 Ulysses，各卡计算两种 attention 分支。

图片参考模式是官方 **Ref2VA-like**：FL2VA 权重接收参考图，不是 MiniMax 的独立 Ref2VA transformer。默认开启 RDT 0.25 的 DBCache 近似跨步缓存，支持逐请求关闭；暂不接入 LightX2V 或整块 DiT 编译；`inference_kernels` 控制官方融合/局部编译内核组合。

## 一条命令启动

已部署的服务器更新：

```bash
cd /root/ref2va && git pull --ff-only && bash deploy.sh --gpus 4
```

首次部署、已有机器更新、从镜像创建的新机器，都使用同一入口：

```bash
bash deploy.sh --gpus 4   # 自动识别型号，两套四卡服务，8188 / 8189
bash deploy.sh            # 自动识别型号，一套八卡服务，8188
```

启动命令先离线检查固定版本源码、Diffusers 补丁、Python 环境、依赖版本/导入路径、模型版本及分片；缺什么才安装/下载，完整镜像不重新下载依赖和权重。代码和配置未变且服务仍在运行时复用现有进程；代码、参数或机器启动身份变化时自动重载。**命令会等到每套模型完成预热且对应 API 真正就绪才成功返回**，之后服务继续在后台运行。进程启动或模型就绪但端口尚不可用，不算成功。

需要 Linux x86_64、系统 `python3`、8 张同型号完整 H200/B200/B300、支持当前 CUDA 12.9 PyTorch 的驱动、NVLink/NCCL 和约 250 GB 磁盘空间。依赖/权重检查不等于校验全部权重文件的 SHA-256；CUDA/NCCL 和模型预热检查在目标机器实际执行。

H200 Flex 后端默认使用 **2048 间隔的无屏蔽前缀分桶**（B200/B300 decomposed 后端不补齐）：补齐 token 参与 attention，可能改变生成结果；新双流组合的 H200 耗时仍需复测。已有环境变量会继续生效；如需覆盖旧的 0/1024 设置，用 `REF2VA_TOKEN_BUCKET=2048 bash deploy.sh restart`；`REF2VA_TOKEN_BUCKET=0 bash deploy.sh start` 可回到不补齐的原生布局。

启动依次执行：

1. 自动检查依赖和模型；需要更新时停止本目录旧服务，已满足相同配置的健康实例直接复用。
2. **自动停止本实例所选 GPU 上已有的计算应用**，释放显存。识别到 systemd 应用服务或 Docker 容器时停止其服务/容器；其他情况停止推理进程树，先 TERM，再在超时后 KILL。不会卸载应用或永久禁用服务。清理动作写入 `.runtime/gpu-cleanup.json`。若外部调度器持续拉起应用，启动报错，不会无限杀进程。
3. 检查模型、所选 GPU 型号/卡数与 NCCL。
4. 每卡加载一份 DiT、默认一份视频 VAE；音频 VAE 位于本实例 rank 0。Qwen3-VL 通过 Accelerate 分配到本实例的 GPU，单卡权重预算八卡时 12 GiB、四卡时 24 GiB，禁止 CPU/磁盘权重卸载。两套四卡 worker 各自加载完整模型。
5. 使用一张合成参考图完成条件编码、正式 8 NFE、音视频解码及 MP4 编码预热。默认预热 10 秒、9:16、短边 768。
6. 每套 worker 预热成功后才开放对应 ComfyUI/API 端口，默认 8188；四卡模式还开放 8189。

这是专用八卡服务的启动行为，会中止这些卡上原有的生成任务。设置 `REF2VA_CLEAR_GPU_APPS=0` 可关闭自动清理，改为显存不足时直接退出。清理只在启动执行；生成期间不会停止其他应用。

模型全程常驻。后续生成复用 DiT、Qwen3-VL 和 VAE，不重新读权重，不执行额外去噪预热。同一 prompt、参考图内容、参考尺寸和模型版本命中条件缓存时，也会跳过编码。新序列长度/尺寸仍可能触发内核编译；启动预热不能覆盖所有输入形状。保留磁盘编译缓存，并在有限数量的新形状后重置 Dynamo 编译图记录，避免官方单次推理实现达到重编译上限。

首次就绪等待期间 Ctrl-C 会取消启动并回收此次实例；命令成功返回后服务留在后台，使用 `bash deploy.sh stop` 停止。取消正在推理的任务会终止整组 NCCL worker，并自动重新加载预热，期间生成接口返回 503。GPU OOM、CUDA/NCCL 错误、rank 退出、启动超时及推理卡死会自动回收整组 worker，重新加载模型并完成启动预热。UI 和查询接口在恢复期间保留，生成接口返回 503，就绪后自动恢复服务。

## 后台运行与 worker 自动恢复

```bash
bash deploy.sh --gpus 4 # 检查/安装/必要时重载，等两套 API 就绪后返回
bash deploy.sh status   # 各实例 running、worker_ready、ready、阶段及恢复状态
bash deploy.sh logs     # 持续查看服务日志；Ctrl-C 只退出日志查看
bash deploy.sh stop     # 停止全部实例的控制进程、UI 和 GPU worker
```

旧的 `deploy/start/up/restart` 仍可使用，均进入统一流程，不再需要区分安装和重启。启动失败直接输出 service/worker 日志并返回非零，回收本次启动的实例；不会仅留下“后台启动成功”提示。默认首次就绪等待上限为 `REF2VA_STARTUP_TIMEOUT + 300` 秒（默认 3900 秒），可用 `--wait-timeout` 或 `REF2VA_LAUNCH_TIMEOUT` 调整。模型就绪但某个端口被其他服务占用/未响应也不能返回成功。

成功返回后，关闭终端或断开 SSH 不影响服务。当前不安装开机 systemd 单元；镜像机器开机后运行同一命令即可，控制进程被 SIGKILL 后也可用同一命令恢复。

恢复会保留磁盘编译缓存，重新加载模型及执行原有基础预热。连续失败按 5、10、20、40、60 秒退避重试，之后每次最多等待 60 秒，稳定运行 300 秒后重置退避。重启期间仅清理本服务的旧进程组，不重复执行启动时的其他 GPU 应用清理；显存仍被其他应用占用时，记录原因并退避等待。

失败请求保留错误和原始日志，**不会自动重跑**；已排队但在恢复期间开始执行的请求也可能失败，客户端需在恢复后决定是否重新提交。`/openvdn/health` 的 `supervision` 包含重启次数、最近错误、退避重试时间和超时配置；每次旧 worker 的错误快照在 `.runtime/backend/failures/`，推理详细日志在 `.runtime/backend/worker.log`。

默认启动超时 3600 秒，单次 worker GPU 阶段请求上限 1800 秒（含输出缓冲等待、条件编码、首次编译、推理、VAE 和像素回传，不含后续纯 CPU 输出尾段）。空闲时每个 rank 主循环报告心跳，连续 60 秒无心跳触发重启；**不根据 GPU 利用率为 0 判断故障**。推理期间使用请求总超时，不用空闲心跳阈值，避免把正常编译误判为卡死。按需调整，例如：

```bash
REF2VA_REQUEST_TIMEOUT=900 REF2VA_STARTUP_TIMEOUT=3600 bash deploy.sh restart
```

| 环境变量 | 默认秒数 | 含义 |
| --- | --- | --- |
| `REF2VA_STARTUP_TIMEOUT` | 3600 | 每次模型加载和预热总时限 |
| `REF2VA_REQUEST_TIMEOUT` | 1800 | 单次 GPU 阶段请求总时限，不含排队/图片下载及纯 CPU 输出尾段 |
| `REF2VA_IDLE_TIMEOUT` | 60 | 空闲 rank 心跳超时 |
| `REF2VA_RESTART_DELAY` | 5 | 首次失败后的重试间隔 |
| `REF2VA_RESTART_MAX_DELAY` | 60 | 连续失败退避的最大间隔 |
| `REF2VA_STABLE_SECONDS` | 300 | 重置失败退避前需保持就绪的时间 |

## 选择 GPU 型号、卡数与端口

```bash
bash deploy.sh                                        # 自动识别型号，一套八卡
bash deploy.sh --gpus 4                               # 自动识别型号，两套四卡
bash deploy.sh --gpu-type b300 --gpus 4                # 显式要求 B300
bash deploy.sh --gpu-type b200 --gpus 4 --port 9000    # 两套四卡 B200，9000 / 9001
```

`--gpus` 是**每个 worker 的卡数**，部署机器仍需提供 8 张 GPU。四卡模式启动两套独立服务：worker-0 使用 GPU 0–3，worker-1 使用 GPU 4–7。指定 `CUDA_VISIBLE_DEVICES=...` 时须列出八个不同设备，按前四/后四拆分。两套服务各有一个 API 端口、一条队列、一套常驻模型；提交、查询、下载同一任务须使用同一端口，不自动负载均衡。两边都有任务时可同时生成两条视频。

| GPU / 每 worker 卡数 | 默认 softmax backend | 默认布局 |
| --- | --- | --- |
| H200 / 8 | flex | ranks=0，双流 Ulysses |
| H200 / 4 | flex | ranks=0，双流 Ulysses |
| B200 / 8 | decomposed | ranks=0，双流 Ulysses |
| B200 / 4 | decomposed | ranks=0，双流 Ulysses |
| B300 / 8 | decomposed | ranks=0，双流 Ulysses |
| B300 / 4 | decomposed | ranks=0，双流 Ulysses |

默认使用本次 B300 对比中最快的组合：双流 Ulysses、RDT 0.25、VAE 4 块合批＋编译；完整参数见 [业务接口](docs/business-api.md)。4×B300 的 10 秒、输出/参考图短边 768 案例热态中位数为 DiT 10.30 秒、视频 VAE 1.88 秒、worker 12.51 秒。该耗时只代表已测案例，不代表各机型均已测得最优布局。请求级 `softmax_ranks` 范围为 0 到单 worker 卡数减 1；0 使用标准 Ulysses。B200/B300 的 decomposed 路径不使用 Flex 的 token 分桶。显式 `REF2VA_SOFTMAX_BACKEND` / `REF2VA_SOFTMAX_RANKS` 仍覆盖默认；从八卡切为四卡时不要保留越界的 rank 数，启动器会在停止旧服务前检查配置。H200 默认关闭 NVLS 以兼容既有主机；B200/B300 使用 NCCL 默认，仍可显式设置 `NCCL_NVLS_ENABLE`。

四卡实例状态、GPU 锁、条件缓存、编译缓存、用户数据库、临时目录和日志分别位于 `.runtime/instances/worker-0/`、`.runtime/instances/worker-1/`。模型文件与唯一文件名的输出目录共用。`status/logs/stop` 自动读取 `.runtime/fleet.json`，无需再传卡数；切换拓扑仍用同一命令并传入新的 `--gpus`。一组 OOM/卡死只重启该组，另一组继续工作。API 健康结果的 `hardware` 返回实际型号、设备列表和单 worker 卡数。

界面中选择带 `_h200_4_fast_v1`、`_b200_4_fast_v1`、`_b300_4_fast_v1` 等后缀的 starter workflow，参数会匹配对应实例，已有用户保存的工作流不会被覆盖。B300 严格校验 `CC 10.3` 和完整显存配置，不把它伪装为 B200；误传 `--gpu-type b200` 会在加载前提示改用 B300/auto。[NVIDIA 型号与计算能力表](https://developer.nvidia.com/cuda/gpus)

四卡/八卡 B300 已完成目标服务器测试；H200/B200 保留对应后端与硬件校验，更新后的组合仍需在对应硬件复测。

### 镜像迁移

- GPU 型号和所选卡的 UUID 在每次启动重新查询，不用镜像里保存的 UUID；默认用本机 0–7，显式 `CUDA_VISIBLE_DEVICES` 必须属于新机器。
- PID 记录绑定机器/boot ID 和进程创建时间，旧机器的 ready、PID、命令不会当成新机器的服务，也不自动重跑镜像中的未完成请求。模型、历史输出和可复用缓存保留。
- 默认同时监听 `0.0.0.0,::`（IPv4 + IPv6）；生成/查询的视频 URL 按当前 HTTP 请求 origin 生成，IPv6 地址保留方括号，不在代码或启动配置中写死公私网 IP。旧 fleet 文件里的 `PUBLIC_BASE_URL` 不会重放。镜像中不要通过 shell/systemd 额外导出旧 IP、旧 GPU UUID 或旧网卡名；这些是用户显式覆盖，程序不会猜测并替换。
- 同机 torchrun 使用 loopback rendezvous 和自动分配端口；两个 worker 不共用固定 master 端口，也不依赖旧主机名。硬件型号/卡数的编译缓存目录分开，B300 不强用 B200 的缓存目录。
- 完整机器镜像可保留 Python 环境和模型；代码目录移动后会检查 venv、editable 包路径和 ComfyUI 节点链接，需要时修复。自定义 `REF2VA_MODELS` 挂载路径仍须在新机器可用。

业务使用稳定反向代理域名时，可显式设置 `PUBLIC_BASE_URL`（单实例）或 `REF2VA_PUBLIC_BASE_URL_0/1`（双实例）；裸 IP 部署建议不设置，让 API 自动按当前请求生成链接。

### IPv4 / IPv6 访问

默认启动命令即可为每个 API 同时开启 IPv4、IPv6，两种地址共用该端口的同一个任务队列。也可显式指定：

```bash
bash deploy.sh --gpu-type b300 --gpus 4 --listen '0.0.0.0,::'
```

`--listen` 优先于 `REF2VA_LISTEN`，接受逗号分隔的 IP 地址（不带端口）。`--listen ::` 只监听 IPv6；`--listen 0.0.0.0` 只监听 IPv4。系统禁用 IPv6 或指定地址不可绑定时会在模型加载前报错，不会悄悄只开启 IPv4。

启动完成前，每个端口都必须通过 `http://127.0.0.1:端口/openvdn/health` 和 `http://[::1]:端口/openvdn/health` 的检查，并确认它们对应当前 worker。显式绑定其他 IP 时检查相应地址。监听配置写入当前实例记录，后续 `bash deploy.sh status` 无需再传环境变量，会显示 `listen` 和 `health_urls`。

外部 IPv6 的 URL 格式是 `http://[服务器IPv6]:8188/...`，另一组为 8189；域名有 AAAA 记录时也可直接使用域名。实例需要配置可路由的 IPv6，子网需有 IPv6 公网路由，安全组和主机防火墙需允许调用方 IPv6 访问 TCP 8188/8189。启动 ready 表示本机 API 可用，不代表已验证公网路由和安全组。

## 连续请求的输出流水线

新旧 REST 接口均在入队前准备图片。每个实例仍串行执行条件编码、DiT、VAE 和像素 GPU→CPU 回传；这些工作结束即释放 GPU 队列，MP4 编码/音频封装的剩余 CPU 工作可以与下一条 GPU 推理重叠。ComfyUI 可视化节点仍等待视频完整写好才返回预览，因此只从 UI 连续提交时，不会提前放行该 UI 队列。

每实例最多保留两条尚未完成的输出，每条像素队列最多四个块；CPU/磁盘持续落后时会施加反压，不无限堆内存，也不能保证任何负载下 GPU 始终 100%。CPU 编码失败只使对应任务失败，不重启正在计算下一条的 GPU。GPU worker 崩溃时，其尚未完成的 CPU 输出也会明确失败；成功结果不会被重写。任务必须等 MP4 和耗时记录全部落盘后才进入 `succeeded`，中间阶段为 `running / encoding_output`。

可用 `REF2VA_PIPELINE_OUTPUT=0 bash deploy.sh restart` 关闭跨请求输出重叠进行对照；默认 `1`。关闭时保持原来的整次 worker 输出完成后释放 GPU 锁；非流式模式的旧 pinned-memory 路径及 `REF2VA_ASYNC_OUTPUT` 开关也继续可用。

耗时 schema 10 新增：

- `gpu_worker_seconds`：条件编码至 VAE、回传、清理及同步结束的墙钟耗时，包含阶段内 CPU 调度/反压，不是纯 GPU kernel 时间。
- `cpu_output_tail_seconds`：GPU 阶段结束后剩余输出处理时间，可与下一任务重叠。
- `output_backpressure_seconds`：GPU 工作前等待空闲输出槽的时间。
- `cross_request_output`：是否启用本次跨请求输出重叠。

`worker_wall_seconds` 包含 GPU 阶段及 CPU 尾段；`generation_wall_seconds`、`processing_wall_seconds` 另包含对应排队/准备范围。单独看 DiT 仍使用 `denoise_seconds`。重叠的编码/回传/VAE 细项不能直接求和。

## 8b200 格式业务接口

已支持 `POST /ic/capcut/edit_gateway/v2/video_generation`、`POST /ic/capcut/edit_gateway/v2/query/video_generation`、`POST /sync_infer`（及业务前缀别名）和 MP4 下载。提交使用 `model/content/resolution/duration/ratio/num_inference_steps/seed/optimization`；图片为 `content[].image_url`，支持 URL / Base64。只传文字即文生视频，`role=reference_image` 为参考图，`first_frame` / `last_frame` 为原生首尾帧条件（不能混用参考图）；返回 `task_id`，查询返回 `task`。文生视频、首尾帧示例和图片预处理规则见 [业务接口文档](docs/business-api.md)。

[完整接口文档与参数说明](docs/business-api.md) · [全参数请求示例（RDT 0.25 / 参考短边 768）](examples/business-request.json)。新接口固定 8 步；省略 seed 时随机，返回实际 seed。旧接口与 UI 行为保持兼容。需要参考短边 512 时只改示例中的 `reference_short_edge`。

## 原 JSON 接口（继续兼容）

提交：`POST /openvdn/jobs`。返回 HTTP 202、`job_id`、`status_url` 和实际输出规格；通过 `GET /openvdn/jobs/{job_id}` 查询。该接口与同实例 UI 共用队列，与 CLI 共用 GPU 文件锁；同实例一次只执行一条 GPU 推理，CPU 输出尾段可与下一条 GPU 推理重叠。

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
| `NCCL_NVLS_ENABLE` | H200=0；B200/B300=NCCL 默认 | 绕过当前 H200 主机的 NVLS multicast 内存绑定 CUDA 401；通信自检和常驻 worker 使用相同值。仅关闭 NVLink SHARP offload，不设置 P2P/NVLink transport 禁用开关；主机修复后可显式设 1/2 重测。 |
| `REF2VA_MODELS` | ./models | 下载、启动使用同一权重目录 |
| `REF2VA_PORT` / `REF2VA_LISTEN` | 8188 / 0.0.0.0,:: | API 端口 / 监听 IP 列表；默认 IPv4 + IPv6，可用 `--listen` 覆盖 |
| `REF2VA_FP8` | 1 | 官方 FP8 线性层；0 为 BF16 |
| `REF2VA_INFERENCE_KERNELS` | 1 | 官方融合/局部编译内核；0 也不代表全部禁用编译 |
| `REF2VA_SOFTMAX_BACKEND` | H200=flex；B200/B300=decomposed | flex / decomposed / ref |
| `REF2VA_SOFTMAX_RANKS` | 0 | Ulysses，各卡处理两分支；默认启用双流 |
| `REF2VA_DUAL_STREAM` | ranks=0 且 inference kernels 开启时为 1 | 显式 0/1 覆盖；非零 ranks 不支持双流 |
| `REF2VA_CACHE_DIT` / `REF2VA_CACHE_DIT_THRESHOLD` | 1 / 0.25 | 默认启用近似残差缓存，可逐请求覆盖 |
| `REF2VA_CACHE_DIT_FN_BLOCKS` / `REF2VA_CACHE_DIT_BN_BLOCKS` | 8 / 8 | 前后完整执行块数 |
| `REF2VA_CACHE_DIT_WARMUP_STEPS` / `REF2VA_CACHE_DIT_LAST_STEPS` | 3 / 1 | 开头/结尾完整执行步数 |
| `REF2VA_CACHE_DIT_MAX_CONSECUTIVE` / `REF2VA_CACHE_DIT_MAX_CACHED_STEPS` | 1 / 2 | 连续/总缓存步数上限 |
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

出现 `Failed to bind NVLink SHARP (NVLS) Multicast memory ... CUDA error 401` 时，失败发生在 NCCL 通信初始化。此部署为该主机默认设置 `NCCL_NVLS_ENABLE=0`，仍必须通过八卡真实 all-reduce 和 all-to-all，不能跳过自检；常驻 worker、取消后的重启沿用同一设置。`/openvdn/health` 的 `nccl.nvls_enable` 和每个结果的 `upstream.parallel.nccl_nvls_enable` 记录实际环境值。它是软件绕行，不代表修复了 Fabric Manager/NVSwitch 状态；性能变化需重新测量。[NVIDIA NVLS 参数说明](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-nvls-enable)

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

视频 VAE 默认每批解码 4 个同形状空间块，并按需编译 decoder 的重复 Transformer block。块只拼在 batch 轴，attention、RoPE、tile 边界、重叠、时间拼接和最终裁剪规则保持原样；FP32 权重和原来的 FP16 autocast 保留。没有扩大 tile、补齐像素或减少计算层数。并行和单 rank VAE 都支持。

业务接口 `optimization` 新增 `vae_tile_batch_size: 1|2|4|8`（默认 4）和 `vae_compile: true|false`（默认 true）；ComfyUI 也有同名控件。`1 + false` 恢复逐块 eager 解码，可做同 seed 对照。启动默认值可用 `REF2VA_VAE_TILE_BATCH_SIZE` / `REF2VA_VAE_COMPILE=0|1` 设置。

启动仍使用逐块 eager 路径执行原有的精确传输/组装校验，**不新增全量编译预热**。首个实际请求才编译其所需的 block 形状，后续内存复用；重启后利用现有 Inductor/Triton 磁盘缓存（仍可能有 tracing/cache loading）。关闭 CUDA graphs 和全量 autotune。每卡首次出现的 batch/shape/stride/dtype/autocast/compile 组合，会把结果与原生逐块 eager 输出比较，必须有限且同时满足 relative L2 ≤ 0.005、max abs ≤ 0.05；失败直接报错，不能当作成功。只缓存最近 64 个数值校验记录，不缓存图像 tensor；Dynamo 重置后重新校验编译路径。这是实现正确性的抽样数值门槛，**不是全片逐像素一致或感知画质保证**；需要目标 GPU 实测速度及同 case 对照。

schema 13 的业务任务结果新增 `video_vae_decode`，包含每卡 tile 数、实际 decoder 调用数、batch 分布、编译命中和数值校验误差。`timings.video_vae_compile_seconds`、`video_vae_tile_decoder_seconds`、`video_vae_tile_stitch_seconds`、`video_vae_verification_seconds` 分别报告跨卡最大值；均是 `video_vae_decode_seconds` 内的诊断子项，不能再累加。DiT 的 `compilation` 字段仍仅统计去噪。首次 VAE 编译/校验请求不能计入热态测速。

同一请求对比逐块 eager、4 块合批、4 块合批 + 编译（每组一次预热、三次交错热跑，保留完整响应）：

```bash
python3 scripts/benchmark_vae.py --server http://HOST:8188 \
  --request-file case.json --output-dir work/vae-benchmark
```

使用原版视频/音频 VAE。输出阶段在 GPU 按 8 帧处理颜色、缩放、uint8 转换，提前裁掉超出目标时长的帧；只复制目标尺寸的 RGB 到 CPU。默认使用两个固定大小的 pinned CPU 缓冲区，独立 CUDA stream 执行像素准备和非阻塞 D2H，CPU 同时编码已完成的批次。消费者只等待该批次的完成事件，编码结束后才复用缓冲区；不在每批前后同步整个 GPU。保留像素运算顺序、舍入、插值、libx264 `veryfast` / CRF 23 / 8 线程和 AAC 参数。MP4 仍原子提交，失败不留下可被误认成功的文件。

`REF2VA_ASYNC_OUTPUT=0` 恢复原有单预取线程输出；CPU 测试也使用此路径。启动合成案例逐批比较异步传回的 RGB 与原同步路径，要求逐元素一致；失败则不开放 UI。`upstream.output_encoding` 返回 `async_pinned_output`、`pixel_timing_method` 和 `pixel_parity`。GPU 对照测试还覆盖非默认生产流、缓冲区复用、非整批尾帧，以及同步/异步输出 MP4 的解码后音视频一致性；在没有 CUDA 的环境中明确跳过，不计为通过。

## 单次请求内的 DiT 去重与同步优化（schema 4）

默认 `REF2VA_EXACT_RUNTIME=1`。显式关闭 DBCache 时完整运行 8 次 DiT，保留所有注意力和线性分支、FP8 设置、权重及采样器，不启用跨步残差缓存：

- 同一次请求的 `RoPE(position_ids)`、`token_refiner(context_embedder(prompt_embeds))` 只计算一次，其余 7 次复用。输入存储、形状、stride、版本、dtype、设备与 autocast 变化会失效；请求结束或异常立即释放。不同请求不共享这些结果。
- 每层线性输出投影的 GPU 布尔索引/`any()`/`sum().item()` 改为 CPU 已知区间切片，保留原 GEMM 的输入形状、连续布局和精度。
- 每步计时使用 CUDA event，循环结束统一读取，去掉仅为计时增加的逐步全设备同步。模型本身需要的同步不变；没有改变完整隐藏特征的 gather 和输出投影次序。

适配层只接受固定 OpenVDN 函数的完整源码 hash，保持 `.deps` 工作区不变；源码或注意力方法不匹配直接拒绝启动。启动及历史预热时，在相同输入上对比缓存常量与重算结果、每个 attention 模块第一次输出投影与原始布尔索引路径，要求有限且逐元素一致。各卡先完成去噪并交换校验状态，再统一报错，避免某一卡在 collective 前退出。记录位于 `.runtime/backend/exact-runtime-parity.json`、`warmup-report.json` 以及每次结果的 `upstream.exact_runtime.by_rank`。这是组件和预热案例校验，不是所有提示词的端到端质量证明；本地 CPU 通过也不代表 H200 已测性能。

两个开关都只在启动配置：需要回退本轮优化时执行 `REF2VA_EXACT_RUNTIME=0 REF2VA_ASYNC_OUTPUT=0 bash deploy.sh start`，其余模型、并行 VAE 和编译缓存配置不变。`/openvdn/health` 返回开关及 `metrics_schema_version: 9`。

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

`softmax_ranks` 现在可逐请求指定：6 为 6+2，5 为 5+3，4 为 4+4，0 为普通八卡 Ulysses。只在串行队列的请求边界切换分工，保留权重与 communicator。不同布局单独记录编译几何；首次切换可能编译，随后复用。服务默认改为 ranks=0 的双流路径；API 仅传非零 ranks 会自动关闭该请求的双流，显式冲突则拒绝。比较脚本不永久更改服务默认。改变并行分工不引入缓存近似，但浮点运算顺序可能不同，不能承诺逐像素一致。

`profile: true` 按请求开启 CUDA event 分析，默认关闭。返回 `metrics.upstream.parallel_profile`：

- `by_rank`：每卡角色、head 数、各段 `total_ms`、`ms_per_nfe`、调用次数。
- 分段包含输入准备、所有 DiT blocks、attention、FFN、末尾 gather、输出 head；分支路径进一步包含 QKV、gate、打包、分发等待、softmax/linear 计算、回传及输出投影。
- `branches` / `max_ms_per_nfe` 用于找慢卡和分支不均衡。计时测到的是计算流上的时间跨度，包含可见等待和提交间隙，并不是独立 NCCL kernel 的纯耗时。
- `branch_dispatch` 包含 `branch_pack` 和 `branch_relevant_wait`，`output_dispatch` 包含 `output_a2a` 和 `output_unpack`，`blocks` 包含 attention/FFN；这些层级有重叠，不能相加，也不能累加八卡计时作为请求耗时。分析会有额外开销，正式测速使用 `profile: false`。

Schema 12 增加按请求的细粒度分析，业务接口通过 `optimization.profile=true` 开启，并在 `task.profiling` 返回完整结果，旧接口的返回位置不变：

- softmax 分解为 dense/window attention、Q/K/V gather、K/V contiguous、scatter 和计划准备。仅实际执行 decomposed 路径时有这些分项，flex/ref 不伪造数据。
- linear 分解为 Q 激活、K/V 空间卷积、时间卷积与激活、A/B 统计、文本状态（含统计与求解）、Cholesky、三角求解、逆矩阵乘积、正反向扫描、状态 gather、query readout、norm/gate。优化路径另外记录 `linear_fused_delta`、`linear_chunk_compose_forward/reverse`、`linear_chunk_scan`。循环按整段计时，不在每帧后同步。
- 各阶段 GPU event 数据在 `by_rank[].total_ms/calls`，对应未同步的 CPU 提交墙钟在 `by_rank[].fine.host_scopes`。后者包含提交、Python 调度及同步等待，不能当成纯 CPU 算术。
- 进一步传 `optimization.profile_kernels=true`（要求同时 `profile=true`），只在 rank 0 和第一个 linear rank 采集 PyTorch/Kineto CPU+CUDA 活动。`by_rank[].fine.kernel_trace` 返回实际设备活动区间并集、kernel 调用数和耗时排名、NCCL kernel 累计活动时间、CPU 算子排名。没有 CUPTI/CUDA 活动时明确标记不可用，不把 CPU 推算成 GPU 耗时。多流重叠的 kernel 累计时间也不是端到端延迟。
- 详细计时会扰动调度，kernel tracing 尤其明显。先同参数跑 1 次预热及 3 次 `profile=false` 测速，再单独分析；不能用分析请求计算加速比例。Profiler 收尾时间独立记录在 `kernel_trace.finalize_seconds`。报告整理和跨 rank 指标收集记录为 `timings.metadata_collection_seconds`，包含 profiler 收尾，两者算入 worker 墙钟但不算入 DiT 去噪。
- 普通请求不进入分析包装；分析包装保留当前选中的原生或加速路径，在成功/失败后恢复。分析本身不改变数学运算、不做降精度或新增近似，默认不采集。

### SGLang 算子与通信适配

参考 [SGLang VDN 实现](https://github.com/sgl-project/sglang/blob/5e9342d16f03621f8f434baca2bc4bbdfa4800c7/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3_vdn.py) 和同版本的 [FP32 delta CUDA kernel](https://github.com/sgl-project/sglang/blob/5e9342d16f03621f8f434baca2bc4bbdfa4800c7/python/sglang/kernels/jit/csrc/diffusion/vdn_delta_factors.cuh)，移植到现有 OpenVDN Ref2VA 常驻 worker。保留参考图、音频和窗口定义，无需安装 SGLang/TVM；来源与 Apache-2.0 许可证在 `openvdn_comfy/_vendor/sglang_vdn/`。

全部参数通过业务请求 `optimization`、旧 REST 顶层、ComfyUI 和 CLI 传入：

| 参数 | 默认 | 行为 |
|---|---|---|
| `fused_delta` | `true` | 把视频/文本的 128×128 SPD 求逆、transition 和 injection 合成一个 FP32 CUDA kernel，替换多次 Cholesky/求解/GEMM。启动在实际 GPU 对 FP64 参考做检查；失败报错，不跳过校验。 |
| `boundary_scan` | `true` | 先合成块内全部帧的仿射变换，再扫描 chunk 边界；保留所有帧的信息。首尾锚帧、尾部不足一个 chunk 均处理；不符合边界条件的窗口回退原扫描。 |
| `fast_softmax` | `true` | decomposed 路径改为 index_select 和连续 Q 切片，保留 FA4/cuDNN 后端及原 mask；有 stride 时仍复制连续，不把错误布局交给 FA4。flex/ref 路径不受此开关影响。 |
| `dual_stream` | `true`（ranks=0） | 双流 Ulysses：共享一次原始 QKV 交换，softmax/linear 两条 CUDA 流重叠，独立回传再汇合。要求 `softmax_ranks=0`、`inference_kernels=true`，支持 4/8 卡。每种新几何的首个 block 与原 Ulysses 进行全 rank 数值核对。 |

`fused_delta` 和 `boundary_scan` 保持方程与全部输入，改变 FP32 运算顺序，不能称为逐位等价；没有新增 K/V 下采样。双流路径的 gate GEMM 形状也会改变。Cache-DiT 和 `linear_kv_keep_ratio` 仍单独控制，做精度对照时设为关闭和 `1.0`。

启动默认值可设 `REF2VA_FUSED_DELTA=0|1`、`REF2VA_BOUNDARY_SCAN=0|1`、`REF2VA_FAST_SOFTMAX=0|1`、`REF2VA_DUAL_STREAM=0|1`；启用最后一项须同时 `REF2VA_SOFTMAX_RANKS=0`。仅预热原配置的一种 shape，其他组合首请求按需编译。融合 kernel 需要 `nvcc`。一键启动先发现兼容的系统/项目编译器；缺失或不完整时，自动下载 NVIDIA 官方 CUDA 12.9.1 的 `cuda_nvcc`、`cuda_cudart` 和 `cuda_cccl` 组件，校验固定 SHA256，安装到 `.runtime/toolchains/cuda-12.9.1/`。首次下载约 84 MB；已有完整工具链离线复用，不更改系统驱动和 Python 包。宿主机需有 `g++`（Amazon Linux/RHEL：`dnf install -y gcc-c++`；Ubuntu：`apt-get install -y g++`）。可通过 `REF2VA_NVCC=/path/to/nvcc` 明确指定编译器。

工具链检查和实际 kernel 编译/加载在停止旧服务之前完成；内核按源码、CUDA toolkit 版本及 GPU 架构缓存到共享 `.runtime/cuda-kernels/`，4+4 两个实例复用同一编译产物。镜像更换 H200/B200/B300 后重新选择对应架构，不依赖 IP。数值正确性仍在 worker 启动时用实际 GPU 检查，不因预编译而跳过。组件来源与 SHA256 见 [NVIDIA CUDA 12.9.1 redistrib](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json)。

返回 `task.optimizations.linear_acceleration.by_rank` 包含实际路径调用次数、求解/扫描 GPU 校验；`task.optimizations.dual_stream.by_rank` 包含双流启用与校验状态。不开 profile 也返回这些记录。双流的 `softmax_return_launch` / `linear_return_launch` 是提交跨度，`output_stream_join` 是主流等待；只有 kernel trace 的 NCCL 活动统计才是设备通信活动时间。第一种新双流几何的数值核对会增加一次原路径 forward，不能混入热态测速。

同 case 对照脚本（请求文件为 `examples/business-request.json` 格式，替换真实参考图 URL，固定 seed）：

```bash
python3 scripts/benchmark_sglang_acceleration.py --server http://HOST:8188 \
  --request-file case.json --output-dir work/sglang-ablation --repeat 3 --kernels
```

默认比较原路径、单独融合求解、单独边界扫描、单独窗口整理、三项合并、普通 Ulysses 串行、普通 Ulysses 双流。每组预热一次，然后交错进行三次无 profile 测速，最后单独采集细粒度分析；`--kernels` 再采集三组代表性 CPU/CUDA trace。可用 `--variants native combined ulysses_dual` 缩减。输出保存逐次请求、响应、视频 URL、`profiling.json` 和 `summary.json`；保留原请求的 Cache-DiT 配置，比较无缓存时在请求中关闭它。热态仍有编译的组不计算速度，服务实例改变立即停止，恢复跑批不会重复提交已接收任务。

当前移植继续使用已有 FP8 精度路径，**未切换 MXFP8**；不能把 SGLang 的 6.9 秒（8×B200、345 帧 T2VA、MXFP8 热态）直接当作这套 Ref2VA 的承诺。实际 GPU 时延与画质需在部署后用同 case 测量；本地 CPU 数值测试不替代 CUDA/视频验证。

### Cache-DiT / DBCache 参数

这是按 [Cache-DiT DBCache 算法](https://github.com/vipshop/cache-dit/tree/main/src/cache_dit/caching/cache_blocks) 独立实现的 OpenVDN 八卡适配层 `openvdn_dbcache_adapter_v1`，不是直接安装其 Python 包或 ComfyUI 原生 H3 插件，不启用 TaylorSeer。先完整计算前 Fn 层，比较前缀残差与上次完整计算时的前缀残差；足够相似时复用中间层残差，再完整计算最后 Bn 层。保留原模型的混合 attention、参考图条件、音频、8 次采样调用和后处理。缓存命中会改变去噪轨迹，效果需逐案例对照。

所有参数都可通过 REST / ComfyUI / CLI 按请求设置，不用重启：

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `cache_dit` | true | 开关；关闭时无残差缓存拷贝或额外决策 collective |
| `cache_dit_threshold` | 0.25 | 复用变化阈值，0–1；越小越保守，0 完全不复用；1 仍需通过变化量检查 |
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
  "softmax_ranks": 0,
  "dual_stream": true,
  "profile": false,
  "cache_dit": true,
  "cache_dit_threshold": 0.25,
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

## Attention、通信与流式输出优化（schema 9）

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
| `linear_kv_keep_ratio` | `1.0` | 可选 `1.0/0.5/0.25`，控制 linear 分支每帧视频 K/V 统计保留的位置比例。`1.0` 走原路径；降低比例是有损近似，可能影响细节、身份和时序一致性，需同 seed 对照。 |
| `isolate_padding` | `false` | 仅在 `softmax_backend=flex`、`attention_kernel=native` 时可用。用 BlockMask 排除中间 gap 的 key，保持完整块的快速路径，不使用 score_mod；full-cover 情况也通过隔离 mask，不激活额外 linear 分支。是否满足低开销目标须实测。 |
| `streaming_output` | `true` | 并行 VAE 边解码边传回 clip；rank 0 按原生时间混合顺序提交有效帧，CPU H.264 与后续解码重叠。编码队列最多 4 个像素块，非零卡最多保留 2 个异步发送；异常会退出线程并删除 partial MP4。关闭时恢复整段解码后编码。 |
| `cleanup_policy` | `adaptive` | 热态请求结束保留 CUDA 空闲内存池。新编译、启动校验、显存压力或每 32 次请求清理；`always` 恢复每次 gc/empty_cache。新的条件编码前所有卡仍释放空闲缓存，给 Qwen 跨卡临时分配腾出空间。 |

服务启动默认值也可通过 `REF2VA_FAST_COMMUNICATION`、`REF2VA_ATTENTION_KERNEL`、`REF2VA_LINEAR_STATS_CHUNK_FRAMES`、`REF2VA_ISOLATE_PADDING`、`REF2VA_STREAMING_OUTPUT`、`REF2VA_CLEANUP_POLICY` 设置。若要完全恢复此前调度，用 `REF2VA_FAST_COMMUNICATION=0 REF2VA_STREAMING_OUTPUT=0 REF2VA_CLEANUP_POLICY=always bash deploy.sh start`。现有 `REF2VA_ASYNC_OUTPUT` 控制非流式输出的 pinned-memory 预取。

返回 `metrics.upstream.optimizations`，包含实际参数、通信小张量/真实 NCCL 一致性检查、attention 选择和清理原因。隔离开启时 `compilation.token_bucket.policy=prefix_gap_isolated_v3`，`padding_attention=excluded_keys`；默认仍为 `prefix_gap_unmasked_v2`。隔离与原生不补齐在有效 key 集合上等价，浮点核/矩阵尺寸不同，不能承诺生成视频逐像素一致。Cache-DiT 仍按原参数独立运行。

`linear_kv_keep_ratio` 可在业务 API 的 `optimization` 内逐请求指定，不需要重启或重载模型。例如，在完整请求中使用：

```json
{
  "optimization": {
    "softmax_ranks": 0,
    "dual_stream": true,
    "linear_kv_keep_ratio": 0.5,
    "profile": false,
    "cache_dit": {"enabled": true, "rdt": 0.25}
  }
}
```

这里的 `softmax_ranks=0`、`dual_stream=true` 同时适用于四卡和八卡 worker。旧 `/openvdn/jobs` 接口把 `linear_kv_keep_ratio` 放在顶层，CLI 使用 `--linear-kv-keep-ratio 0.5`，ComfyUI 节点也有对应选项。省略时默认为 `1.0`，不自动开启近似。完整业务请求见 [examples/business-request.json](examples/business-request.json)。

采样发生在原有特征和卷积计算之后：将每帧展平的空间位置均匀划成 `ceil(S × ratio)` 段，各取一个中点，同一组索引同时作用于视频 K/V/beta；索引不使用随机数。统计矩阵 A、B 均乘 `S/保留数量` 以校正求和尺度，后续扫描仍使用原始 S。完整 Q、读出、文本状态、softmax、投影和通信数据量保持不变，因此 `0.5` 不代表整个 linear 分支或视频生成提速一倍。采样还会增加索引读取成本；速度和画质均需要 GPU 实测。

返回的 `optimizations.linear_kv` 包含请求比例、是否实际执行近似、逐 rank 的原始/保留 token 数与统计调用次数。业务 API 查询结果位于 `task.optimizations.linear_kv`。开启 `optimization.profile=true` 后，`timings` 增加 `linear_kv_select_seconds`、`linear_frame_statistics_seconds`、`linear_kv_rescale_seconds`，均为各 rank 累计 CUDA event 耗时的最大值，已包含在去噪耗时中，不能再相加到总耗时；关闭 profile 时为 `null`。`1.0` 也可开启 profile 测量完整统计阶段作为对照。未执行 linear 的层/rank 不记采样，Cache-DiT 跳过的层也不会虚报调用。

`1.0` 且关闭 profile 时完全旁路采样包装，逐 rank 报告为 `instrumented=false`、`statistics_calls=null`，表示未计数，而非没执行 linear 分支。不同比例单独标记编译几何，新比例首次请求可能编译，后续复用；默认启动仍只预热原路径。请求成功或失败都会恢复原始方法并释放采样索引，避免影响后续 `1.0` 请求。

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
NCCL_NVLS_ENABLE=0 .venv-vdn/bin/torchrun --standalone --nproc_per_node=8 scripts/validate_optimization_kernels.py
```

覆盖 1+7 至 7+1 的 pack/unpack/NCCL，以及 FA4 隔离窗口/full-cover 和 decomposed 对 fp32 dense 参考的容差校验。**本地 CPU 回归通过不等于 H200 新路径已通过；只有服务器校验与热态对照完成后才能报告实际加速。**
