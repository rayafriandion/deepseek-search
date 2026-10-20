# DeepSeek 联网搜索插件

通过 DeepSeek 官方 **Search / Fetch** 接口，为 MaiBot 的 **Planner 与 VLM（视觉 Planner）** 提供联网搜索与网页抓取能力，并通过**普通 / 增强双模式**解决 Planner/VLM 「想不起来用搜索」导致的事实准确性不足问题。

## 功能

- **两个常驻可见工具**（`visibility="visible"`，Planner/VLM 每轮都能直接看到，无需先经内置 `tool_search` 发现）：
  - `deepseek_web_search`：联网搜索，返回标题、链接、摘要、来源、发布时间、权威性（0-4）、相关分（0-3），可按需返回网页正文；支持 `site:`、`-site:`、引号、`lang:`、`after:`、`before:` 等搜索语法；
  - `deepseek_web_fetch`：抓取网页，返回 Markdown 格式标题与正文，用于精读搜索结果原文。
- **双模式**：
  - `normal`（普通，默认）：工具可用，但不额外施压；
  - `enhanced`（增强）：通过 `maisaka.planner.before_request` Hook，在**每次 Planner/VLM 请求前**注入督促指令（系统消息），并**强化两个工具的描述**，迫使 Planner/VLM 积极调用搜索核实事实。注入文本与工具描述催促文本均可在配置中自定义。
- **可配置接口地址**：`search_url` / `fetch_url` 默认指向 DeepSeek 官方接口，可改为代理或任何兼容服务。
- **工程细节**：Bearer 鉴权、请求节流（默认 4 QPS，低于账号级 5 QPS 限制）、408/429/5xx 指数退避重试、OpenAI 兼容错误格式解析、结果按字符数截断、API Key 缺失时的明确报错。
- **命令**：`/dssearch` 系列命令用于模式切换与状态查询（见下文）。

> Planner 与 VLM 在 MaiBot 中复用同一条 `chat_loop_step` 请求链路（VLM 即视觉 Planner），因此工具注册一次即同时覆盖两者，增强模式注入也同时作用于两者。

## 安装

将本目录放入 MaiBot 的 `plugins/` 下即可（当前已就位）：

```text
plugins/deepseek-search/
```

编辑 `config.toml`，至少填写 `[api].api_key`。保存后插件配置会热重载。

## 配置（config.toml）

| 节 | 字段 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `[plugin]` | `enabled` | `true` | 是否启用插件 |
| `[api]` | `api_key` | `""` | **必填**，DeepSeek API Key |
| `[api]` | `search_url` | `https://api.deepseek.com/v1/tools/search` | Search 接口 POST 地址 |
| `[api]` | `fetch_url` | `https://api.deepseek.com/v1/tools/fetch` | Fetch 接口 POST 地址 |
| `[api]` | `timeout_seconds` | `30` | 单次请求超时（秒） |
| `[api]` | `max_retries` | `2` | 可重试错误的最大重试次数 |
| `[api]` | `min_request_interval_seconds` | `0.25` | 请求节流间隔（秒） |
| `[search]` | `default_count` | `10` | 未指定 count 时的默认条数 |
| `[search]` | `max_count` | `20` | 指定 count 时的上限 |
| `[search]` | `need_content` | `false` | 是否默认返回网页正文 |
| `[search]` | `localization` | `""` | 本地化 IP |
| `[search]` | `max_summary_chars` / `max_content_chars` | `1200` / `4000` | 返回给模型的截断长度 |
| `[fetch]` | `max_content_chars` | `12000` | 抓取正文截断长度 |
| `[mode]` | `mode` | `normal` | `normal` 或 `enhanced` |
| `[mode]` | `enhanced_instruction` | 内置督促指令 | 增强模式注入的指令文本 |
| `[mode]` | `enhanced_tool_suffix` | 内置催促文本 | 增强模式附加到工具描述的催促文本 |

## 命令

| 命令 | 作用 |
| --- | --- |
| `/dssearch` 或 `/dssearch status` | 查看模式、接口地址、API Key 状态与调用统计 |
| `/dssearch mode` | 查看当前运行模式 |
| `/dssearch mode normal` | 运行时切换到普通模式 |
| `/dssearch mode enhanced` | 运行时切换到增强模式 |
| `/dssearch help` | 帮助 |

命令级切换是运行时覆盖，优先级高于 `config.toml`；修改 `config.toml` 会清除覆盖并以文件为准。

## 计费与限流（DeepSeek 官方）

- Search / Fetch 均按次计费：每 1,000 次成功请求 5 元；失败、无结果不计费。
- 账号级 QPS 限制：Search 5、Fetch 5（各自独立）；Fetch 另有目标域名级 QPS 限制。
- 官方文档：<https://api-docs.deepseek.com/zh-cn/api/search>、<https://api-docs.deepseek.com/zh-cn/api/fetch>、<https://api-docs.deepseek.com/zh-cn/guides/search>

## 增强模式工作原理

1. Hook `maisaka.planner.before_request`（BLOCKING）在 Planner/VLM 请求模型前触发；
2. 仅当本轮候选工具中包含本插件工具时才生效（避免干扰子代理）；
3. 在请求上下文末尾追加一条系统消息（固定 `item_id`，防重复注入），督促先搜索再回答；
4. 在本插件两个工具的 OpenAI function 描述末尾附加催促文本；
5. 通过 `modified_kwargs` 全量返回修改后的 `items` 与 `tool_definitions`。

## 开发说明

- 本插件不修改 MaiBot 本体任何文件，仅通过 `maibot_sdk` 与主程序交互；
- Runner 与主进程隔离，插件异常不影响主程序；
- 接口封装见 `deepseek_client.py`，主逻辑见 `plugin.py`；
- 本地测试：`python _local_test.py`（91 项断言，覆盖组件收集、配置模型、格式化、错误重试、Hook 注入与防重、命令、工具参数校验、生命周期，无需真实 API Key）；
- 仓库地址：<https://github.com/rayafriandion/deepseek-search>。
