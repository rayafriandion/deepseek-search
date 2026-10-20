"""DeepSeek 联网搜索插件 - 主入口。

通过 DeepSeek Search / Fetch 接口向 MaiBot 的 Planner 与 VLM（视觉 Planner）
暴露两个常驻可见工具：

- ``deepseek_web_search``：联网搜索，返回结构化结果；
- ``deepseek_web_fetch``：抓取网页，返回 Markdown 正文。

两种运行模式：

- ``normal``（普通）：工具对 Planner/VLM 可见可用，但不额外施压；
- ``enhanced``（增强）：通过 ``maisaka.planner.before_request`` Hook 在每次
  Planner/VLM 请求前注入督促指令，并强化两个工具的描述，迫使 Planner/VLM
  积极调用搜索核实事实，避免因不调用导致的事实准确性不足。

Planner 与 VLM 在 MaiBot 中走同一条 ``chat_loop_step`` 请求链路，因此
``@Tool`` 注册一次即可同时覆盖两者，Hook 注入也同时作用于两者。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from maibot_sdk import (
    CONFIG_RELOAD_SCOPE_SELF,
    Command,
    Field,
    HookHandler,
    MaiBotPlugin,
    PluginConfigBase,
    Tool,
)
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder, ToolParameterInfo, ToolParamType

try:
    from .deepseek_client import (
        DEFAULT_FETCH_URL,
        DEFAULT_SEARCH_URL,
        DeepSeekAPIError,
        DeepSeekClient,
        format_fetch_result,
        format_search_results,
    )
except ImportError:
    from deepseek_client import (  # type: ignore[no-redef]
        DEFAULT_FETCH_URL,
        DEFAULT_SEARCH_URL,
        DeepSeekAPIError,
        DeepSeekClient,
        format_fetch_result,
        format_search_results,
    )

# 运行模式常量
MODE_NORMAL = "normal"
MODE_ENHANCED = "enhanced"

# 工具名称（Hook 中按名称识别并强化描述）
SEARCH_TOOL_NAME = "deepseek_web_search"
FETCH_TOOL_NAME = "deepseek_web_fetch"

# 增强模式注入消息使用的固定 item_id，用于避免同一轮请求内重复注入
ENHANCED_REMINDER_ITEM_ID = "deepseek_search_enhanced_reminder"

# /dssearch 命令 pattern；命名捕获组通过 matched_groups 传入处理器
COMMAND_PATTERN = r"^/dssearch(?:\s+(?P<action>help|status|mode))?(?:\s+(?P<mode>normal|enhanced))?\s*$"

# DeepSeek 接口对查询长度的硬性限制
QUERY_MAX_CHARS = 128
# DeepSeek 接口对 URL 长度的硬性限制
URL_MAX_CHARS = 8192
# DeepSeek 接口对 count 的硬性限制
COUNT_MIN = 1
COUNT_MAX = 50

SEARCH_TOOL_DESCRIPTION = (
    "联网搜索网页内容。当需要查询实时信息、新闻事件、价格、版本发布、人物近况、数据统计，"
    "或任何你不确定、可能过时的知识时使用本工具，不要仅凭记忆回答。"
    "支持 site:、-site:、引号精确匹配、lang:、after:、before: 等搜索语法。"
    "返回每条结果的标题、链接、摘要、来源站点、发布时间、权威性（0-4）与相关分（0-3），"
    "可按需返回网页正文。"
)

FETCH_TOOL_DESCRIPTION = (
    "抓取指定网页并以 Markdown 返回页面标题与正文。"
    "当搜索结果的摘要不足以确认信息、或用户直接提供了链接需要读取原文时使用本工具。"
)

MISSING_API_KEY_ERROR = (
    "DeepSeek API Key 未配置：请编辑 plugins/deepseek-search/config.toml，"
    "在 [api] 节填写 api_key 后重载插件。"
)


# ===== 配置模型 =====


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__ = "插件"
    __ui_icon__ = "package"
    __ui_order__ = 0

    enabled: bool = Field(default=True, description="是否启用插件")
    config_version: str = Field(default="1.0.0", description="配置版本")


class ApiSectionConfig(PluginConfigBase):
    """DeepSeek 接口配置。"""

    __ui_label__ = "接口"
    __ui_icon__ = "plug"
    __ui_order__ = 1

    api_key: str = Field(default="", description="DeepSeek API Key（Bearer 鉴权，必填）")
    search_url: str = Field(
        default=DEFAULT_SEARCH_URL,
        description="Search 接口 POST 地址，默认为 DeepSeek 官方接口，可改为代理或兼容服务地址",
        json_schema_extra={"label": "Search 接口地址"},
    )
    fetch_url: str = Field(
        default=DEFAULT_FETCH_URL,
        description="Fetch 接口 POST 地址，默认为 DeepSeek 官方接口，可改为代理或兼容服务地址",
        json_schema_extra={"label": "Fetch 接口地址"},
    )
    timeout_seconds: float = Field(default=30.0, description="单次请求超时时间（秒）")
    max_retries: int = Field(default=2, description="遇到 408/429/5xx 或网络错误时的最大重试次数")
    min_request_interval_seconds: float = Field(
        default=0.25,
        description="两次请求的最小间隔（秒）。DeepSeek 账号级 QPS 限制为 5，默认约 4 QPS 留出余量",
    )


class SearchSectionConfig(PluginConfigBase):
    """搜索行为配置。"""

    __ui_label__ = "搜索"
    __ui_icon__ = "search"
    __ui_order__ = 2

    default_count: int = Field(default=10, description="Planner/VLM 未指定 count 时的默认返回条数（1-50）")
    max_count: int = Field(default=20, description="Planner/VLM 指定 count 时的上限截断值（1-50）")
    need_content: bool = Field(default=False, description="是否默认返回网页正文（会增加返回体积）")
    localization: str = Field(default="", description="本地化 IP，留空则不指定")
    max_summary_chars: int = Field(default=1200, description="返回给模型的单条摘要最大字符数")
    max_content_chars: int = Field(default=4000, description="返回给模型的单条网页正文最大字符数")


class FetchSectionConfig(PluginConfigBase):
    """抓取行为配置。"""

    __ui_label__ = "抓取"
    __ui_icon__ = "file-text"
    __ui_order__ = 3

    max_content_chars: int = Field(default=12000, description="返回给模型的抓取正文最大字符数")


class ModeSectionConfig(PluginConfigBase):
    """运行模式配置。"""

    __ui_label__ = "模式"
    __ui_icon__ = "zap"
    __ui_order__ = 4

    mode: Literal["normal", "enhanced"] = Field(
        default="normal",
        description="运行模式：normal=普通（不施压）；enhanced=增强（每次 Planner/VLM 请求都注入督促指令）",
    )
    enhanced_instruction: str = Field(
        default=(
            "【联网搜索增强模式已开启】你必须积极使用 deepseek_web_search 与 deepseek_web_fetch 工具核实事实：\n"
            "1. 涉及实时信息、新闻、价格、版本发布、人物近况、数据统计，或任何你不确定、可能过时的细节时，"
            "先搜索再回答，严禁仅凭记忆作答；\n"
            "2. 搜索摘要不足以确认时，用 deepseek_web_fetch 抓取结果链接的原文再下结论；\n"
            "3. 只有在纯闲聊、情感陪伴等无需事实的场景下才可以跳过搜索；\n"
            "4. 搜索失败时如实告知用户，不要编造搜索结果或链接。"
        ),
        description="增强模式下注入到 Planner/VLM 上下文的指令文本",
    )
    enhanced_tool_suffix: str = Field(
        default="【增强模式】涉及事实、实时信息或任何你不确定的内容时，请优先调用本工具联网核实，不要仅凭记忆回答。",
        description="增强模式下附加到两个工具描述末尾的催促文本",
    )


class DeepSeekSearchPluginConfig(PluginConfigBase):
    """插件完整配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    api: ApiSectionConfig = Field(default_factory=ApiSectionConfig)
    search: SearchSectionConfig = Field(default_factory=SearchSectionConfig)
    fetch: FetchSectionConfig = Field(default_factory=FetchSectionConfig)
    mode: ModeSectionConfig = Field(default_factory=ModeSectionConfig)


# ===== 插件主体 =====


class DeepSeekSearchPlugin(MaiBotPlugin):
    """DeepSeek 联网搜索插件。"""

    config_model = DeepSeekSearchPluginConfig

    def __init__(self) -> None:
        """初始化插件内部状态。"""

        super().__init__()
        self._client: Optional[DeepSeekClient] = None
        # 通过 /dssearch 命令进行的运行时模式覆盖；None 表示以 config.toml 为准
        self._mode_override: Optional[str] = None
        # 简单的内存计数器，用于 /dssearch status 展示
        self._stats: Dict[str, int] = {
            "search_calls": 0,
            "search_failures": 0,
            "fetch_calls": 0,
            "fetch_failures": 0,
            "enhanced_injections": 0,
        }
        self._missing_key_warned = False

    # ===== 生命周期 =====

    async def on_load(self) -> None:
        """插件加载：按配置构造 API 客户端。"""

        self._client = self._build_client()
        if not self._client.configured:
            self._missing_key_warned = True
            self.ctx.logger.warning(
                "DeepSeek 联网搜索插件已加载，但未配置 API Key；"
                "请编辑 plugins/deepseek-search/config.toml 的 [api].api_key"
            )
        self.ctx.logger.info(
            "DeepSeek 联网搜索插件已加载，当前模式: %s（search=%s, fetch=%s）",
            self.current_mode,
            self.config.api.search_url,
            self.config.api.fetch_url,
        )

    async def on_unload(self) -> None:
        """插件卸载：释放 HTTP 会话。"""

        client = self._client
        self._client = None
        if client is not None:
            await client.close()

    async def on_config_update(self, scope: str, config_data: Dict[str, Any], version: str) -> None:
        """处理配置热更新。

        自身配置（config.toml）变化时重建客户端，并清除命令级的运行时模式覆盖，
        使 config.toml 重新成为模式的唯一来源。

        Args:
            scope: 配置变更范围（self/bot/model）。
            config_data: 最新配置数据。
            version: 配置版本号。
        """

        if scope != CONFIG_RELOAD_SCOPE_SELF:
            return
        old_mode = self.current_mode
        old_client = self._client
        self._mode_override = None
        self._client = self._build_client()
        if old_client is not None:
            await old_client.close()
        if not self._client.configured:
            self._missing_key_warned = True
            self.ctx.logger.warning("DeepSeek 联网搜索插件配置已更新，但 API Key 仍为空")
        self.ctx.logger.info(
            "DeepSeek 联网搜索插件配置已更新: version=%s, 模式 %s -> %s",
            version,
            old_mode,
            self.current_mode,
        )

    # ===== 内部辅助 =====

    def _build_client(self) -> DeepSeekClient:
        """按当前强类型配置构造 API 客户端。

        Returns:
            DeepSeekClient: 配置完成的客户端实例。
        """

        api_config = self.config.api
        return DeepSeekClient(
            api_key=api_config.api_key,
            search_url=api_config.search_url,
            fetch_url=api_config.fetch_url,
            timeout_seconds=api_config.timeout_seconds,
            max_retries=api_config.max_retries,
            min_request_interval_seconds=api_config.min_request_interval_seconds,
        )

    @property
    def current_mode(self) -> str:
        """当前运行模式：命令级覆盖优先，其次取配置。"""

        return self._mode_override or self.config.mode.mode

    def _ensure_client(self) -> Optional[DeepSeekClient]:
        """确保客户端存在且已配置 API Key。

        Returns:
            Optional[DeepSeekClient]: 可用客户端；未配置 API Key 时返回 None。
        """

        client = self._client
        if client is None:
            # 极端情况下（配置注入早于 on_load）按当前配置补齐
            client = self._build_client()
            self._client = client
        if not client.configured:
            if not self._missing_key_warned:
                self._missing_key_warned = True
                self.ctx.logger.warning("DeepSeek API Key 未配置，工具调用将直接返回配置错误")
            return None
        return client

    def _resolve_count(self, raw_count: Any) -> int:
        """解析搜索条数入参。

        无法解析时回退到配置默认值，并始终收敛到
        ``[1, min(50, [search].max_count)]`` 区间。

        Args:
            raw_count: Planner/VLM 传入的原始 count（可能为 None 或字符串）。

        Returns:
            int: 合法的返回条数。
        """

        search_config = self.config.search
        upper = min(COUNT_MAX, max(COUNT_MIN, int(search_config.max_count)))
        default = max(COUNT_MIN, min(upper, int(search_config.default_count)))
        if raw_count is None or raw_count == "":
            return default
        try:
            value = int(raw_count)
        except (TypeError, ValueError):
            return default
        return max(COUNT_MIN, min(upper, value))

    @staticmethod
    def _validate_url(raw_url: str) -> Optional[str]:
        """校验抓取 URL 是否满足 DeepSeek Fetch 接口要求。

        Args:
            raw_url: 原始 URL 入参。

        Returns:
            Optional[str]: 不合法时的错误描述；合法时返回 None。
        """

        url = str(raw_url or "").strip()
        if not url:
            return "url 参数不能为空"
        if len(url) > URL_MAX_CHARS:
            return f"url 长度超过 {URL_MAX_CHARS} 字符限制"
        if not re.match(r"^https?://", url, flags=re.IGNORECASE):
            return "url 必须为 HTTP/HTTPS 协议"
        if "@" in url:
            return "url 不允许携带用户名或密码"
        return None

    # ===== Tool：联网搜索 =====

    @Tool(
        SEARCH_TOOL_NAME,
        description=SEARCH_TOOL_DESCRIPTION,
        parameters=[
            ToolParameterInfo(
                name="query",
                param_type=ToolParamType.STRING,
                description="搜索词，1-128 个字符，支持 site:、-site:、引号、lang:、after:、before: 等语法",
                required=True,
            ),
            ToolParameterInfo(
                name="count",
                param_type=ToolParamType.INTEGER,
                description="返回结果条数，1-50；不传则使用插件配置的默认值",
                required=False,
            ),
            ToolParameterInfo(
                name="need_content",
                param_type=ToolParamType.BOOLEAN,
                description="是否同时返回网页正文；不传则使用插件配置的默认值",
                required=False,
            ),
            ToolParameterInfo(
                name="ori_question",
                param_type=ToolParamType.STRING,
                description="触发本次搜索的用户原始问题，有助于搜索引擎理解意图",
                required=False,
            ),
        ],
        # 关键：声明为 visible，使 Planner/VLM 每轮都能直接看到并调用，
        # 无需先通过内置 tool_search 发现，避免"想不起来用"导致事实准确性不足
        visibility="visible",
    )
    async def handle_web_search(
        self,
        query: str = "",
        count: Optional[int] = None,
        need_content: Optional[bool] = None,
        ori_question: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """执行联网搜索并返回面向 LLM 的结构化结果文本。

        Args:
            query: 搜索词。
            count: 返回条数；None 时使用配置默认值。
            need_content: 是否返回网页正文；None 时使用配置默认值。
            ori_question: 触发搜索的用户原始问题。
            **kwargs: 主程序注入的上下文字段（stream_id 等）。

        Returns:
            Dict[str, Any]: 统一工具返回结构（success/content 或 success/error）。
        """

        del kwargs
        client = self._ensure_client()
        if client is None:
            return {"success": False, "error": MISSING_API_KEY_ERROR}

        normalized_query = str(query or "").strip()
        if not normalized_query:
            return {"success": False, "error": "query 不能为空"}
        if len(normalized_query) > QUERY_MAX_CHARS:
            return {"success": False, "error": f"query 长度超过 {QUERY_MAX_CHARS} 字符限制，请精简搜索词"}

        search_config = self.config.search
        resolved_count = self._resolve_count(count)
        resolved_need_content = search_config.need_content if need_content is None else bool(need_content)
        normalized_question = str(ori_question or "").strip()[:QUERY_MAX_CHARS]

        try:
            response = await client.search(
                query=normalized_query,
                count=resolved_count,
                ori_question=normalized_question or None,
                need_content=resolved_need_content,
                localization=search_config.localization,
            )
        except DeepSeekAPIError as exc:
            self._stats["search_failures"] += 1
            self.ctx.logger.warning("DeepSeek 搜索失败: %s", exc)
            return {"success": False, "error": f"搜索失败: {exc}"}

        self._stats["search_calls"] += 1
        content = format_search_results(
            response,
            query=normalized_query,
            max_summary_chars=search_config.max_summary_chars,
            max_content_chars=search_config.max_content_chars,
        )
        return {"success": True, "content": content}

    # ===== Tool：网页抓取 =====

    @Tool(
        FETCH_TOOL_NAME,
        description=FETCH_TOOL_DESCRIPTION,
        parameters=[
            ToolParameterInfo(
                name="url",
                param_type=ToolParamType.STRING,
                description="要抓取的网页地址，须为含 host 的 HTTP/HTTPS URL，不超过 8192 字符",
                required=True,
            ),
        ],
        visibility="visible",
    )
    async def handle_web_fetch(self, url: str = "", **kwargs: Any) -> Dict[str, Any]:
        """抓取网页并返回 Markdown 格式的标题与正文。

        Args:
            url: 目标网页地址。
            **kwargs: 主程序注入的上下文字段。

        Returns:
            Dict[str, Any]: 统一工具返回结构（success/content 或 success/error）。
        """

        del kwargs
        client = self._ensure_client()
        if client is None:
            return {"success": False, "error": MISSING_API_KEY_ERROR}

        url_error = self._validate_url(url)
        if url_error is not None:
            return {"success": False, "error": url_error}
        normalized_url = str(url).strip()

        try:
            response = await client.fetch(url=normalized_url)
        except DeepSeekAPIError as exc:
            self._stats["fetch_failures"] += 1
            self.ctx.logger.warning("DeepSeek 抓取失败: %s", exc)
            return {"success": False, "error": f"抓取失败: {exc}"}

        self._stats["fetch_calls"] += 1
        content = format_fetch_result(
            response,
            url=normalized_url,
            max_content_chars=self.config.fetch.max_content_chars,
        )
        return {"success": True, "content": content}

    # ===== Hook：增强模式督促 =====

    @HookHandler(
        "maisaka.planner.before_request",
        name="deepseek_search_enhanced_push",
        description="增强模式下向 Planner/VLM 注入联网搜索督促指令，并强化搜索/抓取工具描述",
        mode=HookMode.BLOCKING,
        order=HookOrder.LATE,
        timeout_ms=5000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def push_enhanced_mode(self, **kwargs: Any) -> Dict[str, Any]:
        """在 Planner/VLM 请求模型前施加增强模式压力。

        仅在增强模式且本插件工具出现在本轮候选工具中时生效：
        1. 在请求上下文末尾追加一条系统消息，督促积极搜索；
        2. 在本插件两个工具的描述末尾附加催促文本。

        子代理等不携带本插件工具的请求不会被打扰。

        Args:
            **kwargs: Hook 上下文字段（items、tool_definitions 等）。

        Returns:
            Dict[str, Any]: 需要修改时返回 modified_kwargs，否则仅返回 continue。
        """

        if self.current_mode != MODE_ENHANCED:
            return {"action": "continue"}
        client = self._ensure_client()
        if client is None:
            return {"action": "continue"}

        raw_tool_definitions = kwargs.get("tool_definitions")
        tool_definitions = raw_tool_definitions if isinstance(raw_tool_definitions, list) else []
        # 仅当本轮候选工具中包含本插件工具时才注入，避免干扰子代理请求
        owns_tools = any(
            isinstance(definition, dict)
            and self._extract_tool_name(definition) in (SEARCH_TOOL_NAME, FETCH_TOOL_NAME)
            for definition in tool_definitions
        )
        if not owns_tools:
            return {"action": "continue"}

        modified = False

        # 1) 强化工具描述，让模型在工具列表层面就感受到催促
        #    不原地修改传入字典，全部构建新定义，避免对 Hook 载荷产生副作用
        suffix = self.config.mode.enhanced_tool_suffix.strip()
        boosted_definitions: List[Dict[str, Any]] = []
        if suffix:
            for definition in tool_definitions:
                if not isinstance(definition, dict):
                    boosted_definitions.append(definition)
                    continue
                if self._extract_tool_name(definition) not in (SEARCH_TOOL_NAME, FETCH_TOOL_NAME):
                    boosted_definitions.append(definition)
                    continue
                function = definition.get("function")
                if isinstance(function, dict):
                    description = str(function.get("description") or "").strip()
                    if suffix in description:
                        boosted_definitions.append(definition)
                        continue
                    new_function = dict(function)
                    new_function["description"] = f"{description}\n\n{suffix}" if description else suffix
                    new_definition = dict(definition)
                    new_definition["function"] = new_function
                else:
                    description = str(definition.get("description") or "").strip()
                    if suffix in description:
                        boosted_definitions.append(definition)
                        continue
                    new_definition = dict(definition)
                    new_definition["description"] = f"{description}\n\n{suffix}" if description else suffix
                boosted_definitions.append(new_definition)
                modified = True
        tool_definitions = boosted_definitions if modified else tool_definitions

        # 2) 在上下文末尾注入督促指令（系统消息，固定 item_id 防重复）
        raw_items = kwargs.get("items")
        items = raw_items if isinstance(raw_items, list) else []
        instruction = self.config.mode.enhanced_instruction.strip()
        if instruction and not self._has_reminder_item(items):
            reminder_item = {
                "item_type": "SystemMessageItem",
                "meta": {
                    "item_id": ENHANCED_REMINDER_ITEM_ID,
                    "logical_turn_id": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "parts": [{"type": "text", "text": instruction}],
            }
            items = [*items, reminder_item]
            modified = True

        if not modified:
            return {"action": "continue"}

        # Hook 的 modified_kwargs 为全量替换，必须携带全部原始字段
        modified_kwargs = dict(kwargs)
        modified_kwargs["items"] = items
        modified_kwargs["tool_definitions"] = tool_definitions
        self._stats["enhanced_injections"] += 1
        self.ctx.logger.debug(
            "增强模式已注入搜索督促指令: 工具数=%d, 历史条数=%d",
            len(tool_definitions),
            len(items),
        )
        return {"action": "continue", "modified_kwargs": modified_kwargs}

    @staticmethod
    def _extract_tool_name(definition: Dict[str, Any]) -> str:
        """从工具定义中提取工具名，兼容 OpenAI function 与扁平两种结构。

        Args:
            definition: 序列化后的工具定义字典。

        Returns:
            str: 工具名称；结构不符时返回空字符串。
        """

        function = definition.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            return str(name).strip() if isinstance(name, str) else ""
        name = definition.get("name")
        return str(name).strip() if isinstance(name, str) else ""

    @staticmethod
    def _has_reminder_item(items: List[Any]) -> bool:
        """判断上下文中是否已存在增强模式督促消息，避免重复注入。

        Args:
            items: 序列化后的 Context Item 列表。

        Returns:
            bool: 已存在返回 True。
        """

        for item in items:
            if not isinstance(item, dict):
                continue
            meta = item.get("meta")
            if isinstance(meta, dict) and str(meta.get("item_id") or "").strip() == ENHANCED_REMINDER_ITEM_ID:
                return True
        return False

    # ===== Command：模式管理与状态查询 =====

    @Command(
        "dssearch",
        description="DeepSeek 联网搜索插件：模式切换与状态查询",
        pattern=COMMAND_PATTERN,
    )
    async def handle_dssearch_command(self, **kwargs: Any) -> tuple[bool, str, int]:
        """处理 /dssearch 命令。

        用法：
        - ``/dssearch`` 或 ``/dssearch status``：查看当前状态；
        - ``/dssearch mode``：查看当前模式；
        - ``/dssearch mode normal|enhanced``：运行时切换模式；
        - ``/dssearch help``：查看帮助。

        Args:
            **kwargs: 命令上下文字段（matched_groups、stream_id、text 等）。

        Returns:
            tuple[bool, str, int]: (是否成功, 回复文本, 拦截等级)。
        """

        stream_id = str(kwargs.get("stream_id") or "")
        matched_groups = kwargs.get("matched_groups")
        groups: Dict[str, Any] = dict(matched_groups) if isinstance(matched_groups, dict) else {}
        action = str(groups.get("action") or "").strip().lower()
        mode_arg = str(groups.get("mode") or "").strip().lower()

        # 命名捕获组缺失时（如别名匹配）回退到文本解析
        if not action and not mode_arg:
            text = str(kwargs.get("text") or "").strip().lower()
            tokens = text.split()
            if len(tokens) >= 2 and tokens[1] in ("help", "status", "mode"):
                action = tokens[1]
            if len(tokens) >= 3 and tokens[2] in (MODE_NORMAL, MODE_ENHANCED):
                mode_arg = tokens[2]

        if action == "help":
            reply = self._build_help_text()
        elif mode_arg:
            reply = await self._switch_mode(mode_arg)
        elif action == "mode":
            reply = f"当前运行模式: {self.current_mode}"
        else:
            reply = self._build_status_text()

        if stream_id:
            await self.ctx.send.text(reply, stream_id)
        return True, reply, 1

    async def _switch_mode(self, mode: str) -> str:
        """切换运行时模式。

        Args:
            mode: 目标模式（normal/enhanced）。

        Returns:
            str: 面向用户的回复文本。
        """

        if mode not in (MODE_NORMAL, MODE_ENHANCED):
            return f"未知模式: {mode}，仅支持 normal 或 enhanced"
        previous = self.current_mode
        self._mode_override = mode
        if mode == MODE_ENHANCED:
            detail = "后续每次 Planner/VLM 请求都会注入督促指令并强化工具描述"
        else:
            detail = "工具保持可见可用，但不会额外施压"
        self.ctx.logger.info("DeepSeek 联网搜索插件模式切换: %s -> %s", previous, mode)
        return (
            f"已切换运行模式: {previous} -> {mode}\n{detail}\n"
            f"注意：重启或修改 config.toml 后以 [mode].mode = \"{mode}\" 为准，"
            f"当前命令级覆盖优先生效"
        )

    def _build_help_text(self) -> str:
        """构造帮助文本。

        Returns:
            str: 帮助文本。
        """

        return (
            "DeepSeek 联网搜索插件命令帮助：\n"
            "/dssearch - 查看插件状态\n"
            "/dssearch status - 查看插件状态\n"
            "/dssearch mode - 查看当前运行模式\n"
            "/dssearch mode normal - 切换到普通模式（工具可用但不施压）\n"
            "/dssearch mode enhanced - 切换到增强模式（督促 Planner/VLM 积极搜索）\n"
            "/dssearch help - 查看本帮助\n"
            "工具：deepseek_web_search（联网搜索）、deepseek_web_fetch（网页抓取）\n"
            "持久化模式请修改 config.toml 的 [mode].mode"
        )

    def _build_status_text(self) -> str:
        """构造状态文本。

        Returns:
            str: 状态文本。
        """

        client = self._client
        configured = bool(client is not None and client.configured)
        mode = self.current_mode
        mode_note = ""
        if self._mode_override:
            mode_note = f"（命令级覆盖，config.toml 为 {self.config.mode.mode}）"
        return (
            "DeepSeek 联网搜索插件状态：\n"
            f"运行模式: {mode}{mode_note}\n"
            f"API Key: {'已配置' if configured else '未配置（工具将返回配置错误）'}\n"
            f"Search 接口: {self.config.api.search_url}\n"
            f"Fetch 接口: {self.config.api.fetch_url}\n"
            f"搜索调用: 成功 {self._stats['search_calls']} 次 / 失败 {self._stats['search_failures']} 次\n"
            f"抓取调用: 成功 {self._stats['fetch_calls']} 次 / 失败 {self._stats['fetch_failures']} 次\n"
            f"增强模式注入: {self._stats['enhanced_injections']} 次"
        )


def create_plugin() -> DeepSeekSearchPlugin:
    """插件工厂函数，由 Runner 调用。

    Returns:
        DeepSeekSearchPlugin: 插件实例。
    """

    return DeepSeekSearchPlugin()
