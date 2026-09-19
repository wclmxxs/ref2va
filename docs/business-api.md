# 8b200 格式的 Ref2VA 业务接口

业务 JSON 的字段和路由沿用同目录 `8b200` 的网关接口，底层使用当前 ComfyUI 队列和常驻 OpenVDN 8-NFE worker（部署可选八卡或两套四卡）。支持纯文生视频、1–9 张参考图、首帧、尾帧及首尾帧生成；不接受参考视频或参考音频。成功结果通过本服务的 MP4 路由下载。

## 路由

| 用途 | 路由 | 返回 |
| --- | --- | --- |
| 异步提交 | `POST /ic/capcut/edit_gateway/v2/video_generation` | HTTP 200，`{"task_id":"video_<32位hex>"}` |
| 查询 | `POST /ic/capcut/edit_gateway/v2/query/video_generation` | `{"task":{...}}` |
| 同步生成 | `POST /sync_infer` 或 `POST /ic/capcut/edit_gateway/v2/sync_infer` | 等待结束，成功 HTTP 200、失败 HTTP 500，均为 `{"task":{...}}` |
| 视频 | `GET /ic/capcut/edit_gateway/v2/video_generation/{task_id}/content` | MP4，支持 HTTP Range / 拖动播放；任务尚未成功返回 409 |
| 健康与当前部署参数 | `GET /openvdn/health` | 现有健康格式 |

与 8b200 的业务路由一样，这些路由不要求 API Key。现有 `/openvdn/jobs`、ComfyUI、CLI 保持原格式和原默认 seed，不自动迁移历史 job ID。

同步等待默认 1800 秒，可通过启动环境变量 `REF2VA_SYNC_TIMEOUT_SECONDS` 修改（大于 0、最多 86400）。超时返回 HTTP 504 和原 `task_id`；任务继续执行，应查询该 ID，不要重新提交。业务任务与其他入口共享同一实例的生成队列。四卡部署的两个端口各有独立队列，查询和下载必须使用接受该任务的端口。

## 全参数示例

完整请求见 [examples/business-request.json](../examples/business-request.json)。先将示例里的 `image_url.url` 换成真实公网图片 URL，再提交：

```bash
curl -sS http://108.136.40.172:8188/ic/capcut/edit_gateway/v2/video_generation \
  -H 'Content-Type: application/json' \
  --data-binary @examples/business-request.json
```

当前默认采用实测最快组合：`softmax_ranks=0`、`dual_stream=true`、Cache-DiT RDT 0.25、VAE 4 块合批＋编译，开启 fused delta / boundary scan / fast softmax / fast communication / streaming output。保留完整 linear K/V，不开启 profiling。省略整个 `optimization` 即使用这些默认值；显式字段仍可逐请求覆盖。四卡、八卡均使用相同请求格式，型号与卡数通过启动命令指定。

示例使用参考图短边 768、输出短边 768、10 秒 9:16。只将 `reference_short_edge` 改为 `512` 就是 512 参考图组；相同请求发送到 `/sync_infer` 即为同步模式。上面的 IP 是本次测试机，其他机器替换为其当前 IP/DNS，不需要修改代码。

2026-09-19 的 4×B300 同案例三次热态中位数：DiT 10.30 秒、视频 VAE 1.88 秒、worker 12.51 秒、API 13.13 秒。新形状的首次编译、输入下载/条件编码未命中或排队会增加时间，不能将该数值视为所有请求及 H200/B200 的承诺。Cache-DiT 是近似加速；需要完整去噪计算时传 `cache_dit.enabled=false`，其他优化可保持开启。

| 顶层字段 | 含义与约束 |
| --- | --- |
| `model` | 必填，默认模型名 `MiniMax-H3`；同 8b200 接受非空模型别名，但始终执行本服务部署的模型。响应使用 `BUSINESS_MODEL` 环境变量，默认 `MiniMax-H3` |
| `content` | 必填，1–64 个条目；至少一个非空 `text`；图片可省略。多个 text 按顺序用换行连接，总计不超过 24000 字符 |
| `content[].type` | `text` 或 `image_url` |
| `content[].text` | text 条目的提示词 |
| `content[].role` | 图片为 `reference_image`、`first_frame` 或 `last_frame`；text 可省略 role |
| `content[].image_url` | 对象，包含 `url` 和/或 `base64`，规则见下文 |
| `resolution` | 必填，如 `"768P"`、`"704P"`、`"512P"`，大写 P；输出短边为 256–1080 的偶数；仍受总画布面积限制 |
| `duration` | 必填，4–15 秒，可以是小数，按 24 fps 四舍五入为整数输出帧 |
| `ratio` | `"9:16"` 等宽高比，也接受 `"240:427"`；范围 1:4–4:1。省略、null 或 `"adaptive"` 时取第一张图片 EXIF 方向修正后的比例；无图时默认 16:9 |
| `num_inference_steps` | 默认 8，只允许 8；4/6 步与当前 checkpoint 不匹配，返回 400 |
| `seed` | 0–2^63−1 整数；省略或 null 时为每个新任务生成独立的随机 63 位 seed；实际值通过查询 `task.seed` 返回 |
| `reference_short_edge` | 默认 768；128–2048、32 的倍数。只控制 `reference_image` 预处理短边；首尾帧直接使用生成画布，不使用此值 |
| `optimization` | 可选请求级覆盖；未传或字段为 null 时继承服务默认，任务结束后不影响其他请求 |

参考图条目顺序对应 `<Picture 1>`、`<Picture 2>`，不会按文件名或 URL 排序。`adaptive` 只调整画布比例，不把参考图当作首帧；实际输出尺寸、采样尺寸和时长见 `task.render_plan`。

### 文生视频与首尾帧

示例见 [business-t2v-request.json](../examples/business-t2v-request.json) 和 [business-fl2v-request.json](../examples/business-fl2v-request.json)，路由及优化参数完全相同。模式由 `content` 自动决定，无需新增顶层字段：

| 图片角色 | `task.conditioning_mode` | 原生锚点 |
| --- | --- | --- |
| 无图片 | `t2va` | `[]` |
| 1–9 张 `reference_image` | `ref2va_like` | 每张 `ref` |
| 一张 `first_frame` | `i2va` | `["first"]` |
| 一张 `last_frame` | `l2va` | `["last"]` |
| 一张 `first_frame` + 一张 `last_frame` | `fl2va` | `["first", "last"]` |

首尾帧会按首帧、尾帧顺序编码，因此同时提供两张时，`<Picture 1>` 始终是首帧、`<Picture 2>` 始终是尾帧，与 JSON 图片条目顺序无关。不允许重复首/尾帧，也不允许与 `reference_image` 混用。查询返回 `image_anchors`，实际条件元数据也记录原生锚点。

首尾帧按 OpenVDN 原生 keyframe 方式送入 Qwen 与 VAE：第一张缩放到请求的对齐生成画布，第二张等比覆盖缩放并居中裁切到同一画布。建议两张图片使用相同宽高比，并保持 `ratio=adaptive`，避免强制比例造成第一张拉伸或第二张裁切。缓存包含锚点类型及生成画布，不会复用同图的参考图编码或其他尺寸的首尾帧编码。

提示词建议明确两张图对应视频的开始和结束，并描述中间运动，见示例。这是模型原生的首尾帧条件生成，并非输出后粘贴图片，不保证首尾像素完全相同。时长仍按 24 fps 输出；采样帧数按模型支持的 `17n+5` 对齐，多余尾帧沿用现有输出裁切规则。

### 图片 URL / Base64

`image_url` 支持 `{"url":"https://...", "base64":null}`、仅 URL、仅 Base64，或同时传两个字段。非空 Base64 优先，不会访问备用 URL；null、空串或纯空白 Base64 使用 URL。有效 Base64 为纯编码或 `data:image/png;base64,...` / JPEG / WebP Data URI。非空 Base64 损坏时直接失败，不回退 URL。与 8b200 一致，旧式在 `url` 字段传纯 Base64 或 Data URI 也可用。

支持 JPEG、PNG、WebP；每张原始图片最多 20 MiB、4000 万像素，并遵守当前参考图最大边 8192、比例 1:4–4:1 的限制。业务层调用现有 EXIF 修正和 RGB/PNG 归一化后存入内容缓存。HTTP(S) URL 沿用逐连接、逐重定向的公网地址校验；不能读取本地路径。

任务记录只保存内联图片的 SHA-256、大小、格式和尺寸，不保存完整 Base64。图片在排队前完成解析和下载，ComfyUI 队列图只包含任务 ID。请求体上限约 240 MiB（9 张最大 Base64 图片加少量 JSON 开销），超限 HTTP 413。普通 JSON 校验/损坏图片 HTTP 400，网络下载错误 HTTP 502、超时 HTTP 504；这些错误不会创建生成任务。

### 优化参数

以下默认值对应未覆盖的标准部署；查询 `/openvdn/health.request_options` 获取当前运行默认值。

| 字段（位于 `optimization`） | 默认 | 含义 |
| --- | --- | --- |
| `cache_dit.enabled` | true | 请求内的近似跨步缓存 |
| `cache_dit.warmup` | 3 | 开头完整执行的步数，1–8；不是服务启动预热 |
| `cache_dit.rdt` | 0.25 | 复用阈值，0–1；0 禁止复用 |
| `cache_dit.max_continuous_cached_steps` | 1 | 最多连续复用步数，1–7 |
| `cache_dit.fn_blocks` | 8 | 前段完整计算块数，1–49 |
| `cache_dit.bn_blocks` | 8 | 后段完整计算块数，0–49；前后之和必须小于 50 |
| `cache_dit.max_cached_steps` | 2 | 整次生成最多复用步数，0–7 |
| `cache_dit.last_steps` | 1 | 结尾完整计算步数，0–7；与 warmup 之和不超过 8 |
| `attention_kernel` | native | `native` / `decomposed`；不支持 Sol |
| `isolate_padding` | false | 是否隔离补齐 token；只能用于当前 Flex 部署＋native kernel |
| `linear_stats_chunk_frames` | 16 | 8 / 16 / 32 |
| `softmax_ranks` | 0 | 0 到单 worker 卡数减 1；0 为 Ulysses，各卡都处理两分支，默认使用双流 |
| `dual_stream` | true | 两条 CUDA 流重叠计算 softmax / linear，要求 ranks=0 且部署启用 inference kernels |
| `fused_delta` | true | 融合 FP32 delta 求解内核，首次使用进行 GPU 数值校验 |
| `boundary_scan` | true | 合成块内仿射变换后扫描边界，保留全部帧 |
| `fast_softmax` | true | 优化 decomposed window attention 的复制/索引 |
| `linear_kv_keep_ratio` | 1.0 | 1.0 / 0.5 / 0.25；1.0 保留完整视频 K/V，较小值为额外近似 |
| `fast_communication` | true | 已验证的通信优化 |
| `streaming_output` | true | 视频流式输出优化 |
| `cleanup_policy` | adaptive | `adaptive` / `always` |
| `profile` | false | 细粒度 profiling；关闭时仍有常规阶段耗时 |
| `profile_kernels` | false | 需要 profile=true；诊断 CPU/CUDA trace，增加耗时，CUDA 事件依赖有效 CUPTI |
| `vae_tile_batch_size` | 4 | 1 / 2 / 4 / 8；对同形状独立空间 tile 合批；1 + compile=false 恢复逐块 eager |
| `vae_compile` | true | 编译重复 VAE decoder blocks；首次按需编译，不在启动穷举所有形状 |

仅覆盖 `softmax_ranks` 为非零且未显式开启双流时，API 自动关闭本次请求的 `dual_stream`；显式传入非零 ranks 和 dual_stream=true 会返回 400。关闭缓存可传 `{"optimization":{"cache_dit":{"enabled":false}}}`；恢复逐块 VAE 可传 `{"optimization":{"vae_tile_batch_size":1,"vae_compile":false}}`。这些修改只作用于本次请求。

基础字段及四个 Cache-DiT 公共字段沿用 8b200 命名，其余是 ref2va 扩展。FP8、`inference_kernels`、`softmax_backend`、DiT 编译形状容量和 token 桶间隔为部署级配置，不接受新业务请求覆盖；使用健康接口查看，修改需重启。VAE 编译支持上表的请求级开关。未知参数（包括已移除的 Sol 参数）明确返回 400，不静默忽略。

## 查询与耗时

```bash
curl -sS http://108.136.40.172:8188/ic/capcut/edit_gateway/v2/query/video_generation \
  -H 'Content-Type: application/json' \
  -d '{"model":"MiniMax-H3","task_id":"video_<提交返回的32位hex>"}'
```

响应 `task` 的基础字段沿用 8b200：`id`、`model`、`status`、`created_at`、`updated_at`（Unix 秒）、`inference_time_s`、`resolution`、`duration`、`ratio`、`seed`、`task_type=generation`、`modality=video`。另外返回 `num_inference_steps`、`reference_short_edge`、`phase` 和 `render_plan`。

- 状态为 `queued / running / succeeded / failed / cancelled`。服务重启导致未完成任务中断时映射为 cancelled，不自动补交任务。
- GPU 完成但 CPU 输出尚未完成时保持 `running`、`phase=encoding_output`，此时下一条任务可开始 GPU 推理；只有完整输出落盘后才成功。
- 成功后 `task.content.url` 可直接播放/下载；失败在 `task.error` 中给出原因。
- `inference_time_s` 对应 `timings.worker_wall_seconds`，是常驻 worker 的整体处理墙钟耗时，不是纯 DiT 或 GPU kernel 计时。未完成时为 null。
- `task.timings` 保留完整原始耗时：DiT、条件编码/装载、视频/音频 VAE、GPU→CPU、像素准备、H.264、封装、输出整体、清理、排队和总处理等。看约 11 秒的 DiT 指标应使用 `denoise_seconds`。
- `input_prepare_seconds` 为提交阶段的请求读取、校验、图片解析/下载及准备耗时；`api_queue_seconds` 从图片准备完入队算起；`processing_wall_seconds` 包含输入准备和节点执行，排除队列等待。`reference_download_seconds` 为兼容字段，对 Base64 请求也包含输入准备。
- schema 10 新增 `gpu_worker_seconds`（条件编码到 GPU 回传/清理的阶段墙钟）、`cpu_output_tail_seconds`（其后的 CPU 尾段）、`output_backpressure_seconds`（等待输出槽）、`cross_request_output`（跨请求重叠是否启用）。这些时间可能与其他任务重叠；GPU 阶段墙钟也包含 CPU 调度，不是纯 kernel 时间。
- `task.video_vae_decode` 返回 VAE 合批大小、实际 decoder 调用数、各 rank 编译命中及数值校验结果。`timings.video_vae_compile_seconds`、`video_vae_tile_decoder_seconds`、`video_vae_tile_stitch_seconds`、`video_vae_verification_seconds` 是视频 VAE 阶段内的子项，不能再加到该阶段总时间上。
- `task.compilation` 保留 DiT 编译耗时、图复用、分桶布局；`task.cache_dit` 保留实际复用步；`task.optimizations` 返回实际执行的优化和校验结果。编译及输出细分项可能相互包含/重叠，不要机械相加。

反向代理部署时设置 `PUBLIC_BASE_URL=https://你的服务地址`；未设置时根据当前请求的 origin 生成视频 URL。不自动信任 Forwarded 头。双 API 模式若使用反向代理，分别设置 `REF2VA_PUBLIC_BASE_URL_0` 和 `REF2VA_PUBLIC_BASE_URL_1`；否则不要设置全局 `PUBLIC_BASE_URL`，以各请求 origin 生成对应端口的链接。

错误格式为 `{"error":{"type":"invalid_request_error","message":"...","http_code":400}}`。同步超时另含顶层 `task_id`，可以继续查询。任务请求和结果保存在 `.runtime/api/jobs`；四卡实例分别保存在 `.runtime/instances/worker-{0,1}/api/jobs`，可用原内部 UUID 通过旧管理接口排障。

统一启动命令：`bash deploy.sh --gpus 4`，自动识别本机 H200/B200/B300、复用已安装依赖和权重，等待两个 API 就绪后才返回终端。新机器默认无需指定 IP；仍须在对应端口查询该端口提交的任务。
