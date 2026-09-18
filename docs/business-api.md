# 8b200 格式的 Ref2VA 业务接口

业务 JSON 的字段和路由沿用同目录 `8b200` 的网关接口，底层使用当前 ComfyUI 队列和常驻 OpenVDN 8-NFE 八卡 worker。此服务支持 1–9 张 `reference_image`，不接受首尾帧、参考视频或参考音频。未新增 SGLang 代理或 TOS 上传依赖；成功结果通过本服务的 MP4 路由下载。

## 路由

| 用途 | 路由 | 返回 |
| --- | --- | --- |
| 异步提交 | `POST /ic/capcut/edit_gateway/v2/video_generation` | HTTP 200，`{"task_id":"video_<32位hex>"}` |
| 查询 | `POST /ic/capcut/edit_gateway/v2/query/video_generation` | `{"task":{...}}` |
| 同步生成 | `POST /sync_infer` 或 `POST /ic/capcut/edit_gateway/v2/sync_infer` | 等待结束，成功 HTTP 200、失败 HTTP 500，均为 `{"task":{...}}` |
| 视频 | `GET /ic/capcut/edit_gateway/v2/video_generation/{task_id}/content` | MP4，支持 HTTP Range / 拖动播放；任务尚未成功返回 409 |
| 健康与当前部署参数 | `GET /openvdn/health` | 现有健康格式 |

与 8b200 的业务路由一样，这些路由不要求 API Key。现有 `/openvdn/jobs`、ComfyUI、CLI 保持原格式和原默认 seed，不自动迁移历史 job ID。

同步等待默认 1800 秒，可通过启动环境变量 `REF2VA_SYNC_TIMEOUT_SECONDS` 修改（大于 0、最多 86400）。超时返回 HTTP 504 和原 `task_id`；任务继续执行，应查询该 ID，不要重新提交。业务任务与其他入口共享同一生成队列。

## 全参数示例

完整请求见 [examples/business-request.json](../examples/business-request.json)。先将示例里的 `image_url.url` 换成真实公网图片 URL，再提交：

```bash
curl -sS http://43.218.119.131:8188/ic/capcut/edit_gateway/v2/video_generation \
  -H 'Content-Type: application/json' \
  --data-binary @examples/business-request.json
```

相同请求发送到 `/sync_infer` 即为同步模式。示例显式启用 RDT 0.25；只将 `reference_short_edge` 改为 `512` 就是 512 参考图组，输出短边仍为 768。

| 顶层字段 | 含义与约束 |
| --- | --- |
| `model` | 必填，默认模型名 `MiniMax-H3`；同 8b200 接受非空模型别名，但始终执行本服务部署的模型。响应使用 `BUSINESS_MODEL` 环境变量，默认 `MiniMax-H3` |
| `content` | 必填，1–64 个条目；至少一个非空 `text` 和 1–9 个 `image_url`；多个 text 按顺序用换行连接，总计不超过 24000 字符 |
| `content[].type` | `text` 或 `image_url` |
| `content[].text` | text 条目的提示词 |
| `content[].role` | 图片必须是 `reference_image`；text 可省略 role |
| `content[].image_url` | 对象，包含 `url` 和/或 `base64`，规则见下文 |
| `resolution` | 必填，如 `"768P"`、`"704P"`、`"512P"`，大写 P；输出短边为 256–1080 的偶数；仍受总画布面积限制 |
| `duration` | 必填，4–15 秒，可以是小数，按 24 fps 四舍五入为整数输出帧 |
| `ratio` | `"9:16"` 等宽高比，也接受 `"240:427"`；范围 1:4–4:1。省略、null 或 `"adaptive"` 时取第一张参考图 EXIF 方向修正后的比例 |
| `num_inference_steps` | 默认 8，只允许 8；4/6 步与当前 checkpoint 不匹配，返回 400 |
| `seed` | 0–2^63−1 整数；省略或 null 时为每个新任务生成独立的随机 63 位 seed；实际值通过查询 `task.seed` 返回 |
| `reference_short_edge` | 默认 768；128–2048、32 的倍数。控制所有参考图的预处理短边，独立于视频分辨率 |
| `optimization` | 可选请求级覆盖；未传或字段为 null 时继承服务默认，任务结束后不影响其他请求 |

图片条目顺序对应 `<Picture 1>`、`<Picture 2>`，不会按文件名或 URL 排序。`adaptive` 只调整画布比例，不把参考图当作首帧；实际输出尺寸、采样尺寸和时长见 `task.render_plan`。

### 图片 URL / Base64

`image_url` 支持 `{"url":"https://...", "base64":null}`、仅 URL、仅 Base64，或同时传两个字段。非空 Base64 优先，不会访问备用 URL；null、空串或纯空白 Base64 使用 URL。有效 Base64 为纯编码或 `data:image/png;base64,...` / JPEG / WebP Data URI。非空 Base64 损坏时直接失败，不回退 URL。与 8b200 一致，旧式在 `url` 字段传纯 Base64 或 Data URI 也可用。

支持 JPEG、PNG、WebP；每张原始图片最多 20 MiB、4000 万像素，并遵守当前参考图最大边 8192、比例 1:4–4:1 的限制。业务层调用现有 EXIF 修正和 RGB/PNG 归一化后存入内容缓存。HTTP(S) URL 沿用逐连接、逐重定向的公网地址校验；不能读取本地路径。

任务记录只保存内联图片的 SHA-256、大小、格式和尺寸，不保存完整 Base64。图片在排队前完成解析和下载，ComfyUI 队列图只包含任务 ID。请求体上限约 240 MiB（9 张最大 Base64 图片加少量 JSON 开销），超限 HTTP 413。普通 JSON 校验/损坏图片 HTTP 400，网络下载错误 HTTP 502、超时 HTTP 504；这些错误不会创建生成任务。

### 优化参数

以下默认值对应未覆盖的标准部署；查询 `/openvdn/health.request_options` 获取当前运行默认值。

| 字段（位于 `optimization`） | 默认 | 含义 |
| --- | --- | --- |
| `cache_dit.enabled` | false | 请求内的近似跨步缓存 |
| `cache_dit.warmup` | 3 | 开头完整执行的步数，1–8；不是服务启动预热 |
| `cache_dit.rdt` | 0.08 | 复用阈值，0–1；0 禁止复用；示例使用 0.25 |
| `cache_dit.max_continuous_cached_steps` | 1 | 最多连续复用步数，1–7 |
| `cache_dit.fn_blocks` | 8 | 前段完整计算块数，1–49 |
| `cache_dit.bn_blocks` | 8 | 后段完整计算块数，0–49；前后之和必须小于 50 |
| `cache_dit.max_cached_steps` | 2 | 整次生成最多复用步数，0–7 |
| `cache_dit.last_steps` | 1 | 结尾完整计算步数，0–7；与 warmup 之和不超过 8 |
| `attention_kernel` | native | `native` / `decomposed`；不支持 Sol |
| `isolate_padding` | false | 是否隔离补齐 token；只能用于当前 Flex 部署＋native kernel |
| `linear_stats_chunk_frames` | 16 | 8 / 16 / 32 |
| `softmax_ranks` | 6 | 0–7；6 表示 softmax/linear 分支 6+2 分配 |
| `fast_communication` | true | 已验证的通信优化 |
| `streaming_output` | true | 视频流式输出优化 |
| `cleanup_policy` | adaptive | `adaptive` / `always` |
| `profile` | false | 细粒度 profiling；关闭时仍有常规阶段耗时 |

基础字段及四个 Cache-DiT 公共字段沿用 8b200 命名，其余是 ref2va 扩展。FP8、`inference_kernels`、`softmax_backend`、编译开关和桶间隔为部署级配置，不接受新业务请求覆盖；使用健康接口查看，修改需重启。未知参数（包括已移除的 Sol 参数）明确返回 400，不静默忽略。

## 查询与耗时

```bash
curl -sS http://43.218.119.131:8188/ic/capcut/edit_gateway/v2/query/video_generation \
  -H 'Content-Type: application/json' \
  -d '{"model":"MiniMax-H3","task_id":"video_<提交返回的32位hex>"}'
```

响应 `task` 的基础字段沿用 8b200：`id`、`model`、`status`、`created_at`、`updated_at`（Unix 秒）、`inference_time_s`、`resolution`、`duration`、`ratio`、`seed`、`task_type=generation`、`modality=video`。另外返回 `num_inference_steps`、`reference_short_edge`、`phase` 和 `render_plan`。

- 状态为 `queued / running / succeeded / failed / cancelled`。服务重启导致未完成任务中断时映射为 cancelled，不自动补交任务。
- 成功后 `task.content.url` 可直接播放/下载；失败在 `task.error` 中给出原因。
- `inference_time_s` 对应 `timings.worker_wall_seconds`，是常驻 worker 的整体处理墙钟耗时，不是纯 DiT 或 GPU kernel 计时。未完成时为 null。
- `task.timings` 保留完整原始耗时：DiT、条件编码/装载、视频/音频 VAE、GPU→CPU、像素准备、H.264、封装、输出整体、清理、排队和总处理等。看约 11 秒的 DiT 指标应使用 `denoise_seconds`。
- `input_prepare_seconds` 为提交阶段的请求读取、校验、图片解析/下载及准备耗时；`api_queue_seconds` 从图片准备完入队算起；`processing_wall_seconds` 包含输入准备和节点执行，排除队列等待。`reference_download_seconds` 为兼容字段，对 Base64 请求也包含输入准备。
- `task.compilation` 保留编译耗时、图复用、分桶布局；`task.cache_dit` 保留实际复用步；`task.optimizations` 返回实际执行的优化和校验结果。编译及输出细分项可能相互包含/重叠，不要机械相加。

反向代理部署时设置 `PUBLIC_BASE_URL=https://你的服务地址`；未设置时根据当前请求的 origin 生成视频 URL。不自动信任 Forwarded 头。

错误格式为 `{"error":{"type":"invalid_request_error","message":"...","http_code":400}}`。同步超时另含顶层 `task_id`，可以继续查询。任务请求和结果仍保存在 `.runtime/api/jobs`，可用原内部 UUID 通过旧管理接口排障。
