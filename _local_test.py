"""DeepSeek 搜索插件的本地验证脚本（不属于插件运行时文件）。

模拟 Runner 的模块加载方式，验证：
1. 组件收集（2 个 visible Tool、1 个 Command、1 个 HookHandler）；
2. 配置模型默认值与 WebUI Schema；
3. 客户端的格式化、错误解析、重试与节流逻辑（mock HTTP）；
4. 增强模式 Hook 注入逻辑；
5. 命令处理逻辑。
"""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import sys
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

PLUGIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PLUGIN_DIR))

import plugin as plugin_module  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    """记录一条断言结果。"""

    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


class FakeLogger:
    """收集日志的假 logger。"""

    def __init__(self) -> None:
        self.messages: List[str] = []

    def _log(self, message: str) -> None:
        self.messages.append(message)

    def info(self, message: str, *args: Any) -> None:
        self._log("INFO " + (message % args if args else message))

    def warning(self, message: str, *args: Any) -> None:
        self._log("WARN " + (message % args if args else message))

    def error(self, message: str, *args: Any) -> None:
        self._log("ERROR " + (message % args if args else message))

    def debug(self, message: str, *args: Any) -> None:
        self._log("DEBUG " + (message % args if args else message))


class FakeSend:
    """记录发送消息的假 send 能力。"""

    def __init__(self) -> None:
        self.sent: List[str] = []

    async def text(self, text: str, stream_id: str, **kwargs: Any) -> bool:
        self.sent.append(text)
        return True


class FakeContext:
    """假 PluginContext。"""

    def __init__(self) -> None:
        self.logger = FakeLogger()
        self.send = FakeSend()


def load_plugin() -> plugin_module.DeepSeekSearchPlugin:
    """按 Runner 的方式加载插件并注入配置与上下文。"""

    instance = plugin_module.create_plugin()
    config_data = tomllib.loads((PLUGIN_DIR / "config.toml").read_text(encoding="utf-8"))
    instance.set_plugin_config(config_data)
    instance._set_context(FakeContext())
    return instance


def test_components() -> None:
    """验证组件收集结果。"""

    print("== 组件收集 ==")
    instance = load_plugin()
    components = instance.get_components()
    tools = [c for c in components if c.get("type") == "TOOL"]
    commands = [c for c in components if c.get("type") == "COMMAND"]
    hooks = [c for c in components if c.get("type") == "HOOK_HANDLER"]

    check("收集到 2 个 Tool", len(tools) == 2, f"实际 {len(tools)}")
    check("收集到 1 个 Command", len(commands) == 1, f"实际 {len(commands)}")
    check("收集到 1 个 HookHandler", len(hooks) == 1, f"实际 {len(hooks)}")

    tool_by_name = {t["name"]: t for t in tools}
    check("工具名 deepseek_web_search 存在", "deepseek_web_search" in tool_by_name)
    check("工具名 deepseek_web_fetch 存在", "deepseek_web_fetch" in tool_by_name)
    for name, tool in tool_by_name.items():
        visibility = str(tool.get("metadata", {}).get("visibility") or "").lower()
        check(f"{name} 声明 visibility=visible", visibility == "visible", f"实际 {visibility!r}")

    search_tool = tool_by_name.get("deepseek_web_search", {})
    search_params = search_tool.get("metadata", {}).get("parameters", [])
    param_names = {p.get("name") for p in search_params if isinstance(p, dict)}
    check("搜索工具参数包含 query/count/need_content/ori_question",
          {"query", "count", "need_content", "ori_question"} <= param_names, f"实际 {param_names}")

    hook = hooks[0] if hooks else {}
    hook_meta = hook.get("metadata", {})
    check("Hook 目标为 maisaka.planner.before_request",
          hook_meta.get("hook") == "maisaka.planner.before_request", f"实际 {hook_meta.get('hook')!r}")


def test_config_model() -> None:
    """验证配置模型。"""

    print("== 配置模型 ==")
    instance = load_plugin()
    default_config = instance.get_default_config()
    check("默认配置包含全部配置节",
          {"plugin", "api", "search", "fetch", "mode"} <= set(default_config),
          f"实际 {set(default_config)}")
    check("默认模式为 normal", instance.config.mode.mode == "normal")
    check("默认 search_url 为官方接口",
          instance.config.api.search_url == "https://api.deepseek.com/v1/tools/search")
    check("默认 fetch_url 为官方接口",
          instance.config.api.fetch_url == "https://api.deepseek.com/v1/tools/fetch")

    schema = instance.get_webui_config_schema()
    check("WebUI Schema 可生成", isinstance(schema, dict) and bool(schema))

    # 从 config.toml 注入的值生效
    check("config.toml 的 mode 注入为 normal", instance.config.mode.mode == "normal")
    check("config.toml 的 default_count 注入为 10", instance.config.search.default_count == 10)


def test_client_formatting() -> None:
    """验证客户端格式化函数。"""

    print("== 客户端格式化 ==")
    from deepseek_client import format_fetch_result, format_search_results

    search_response = {
        "results": [
            {
                "rerank_score": 2.4,
                "title": "DeepSeek V4 模型发布",
                "url": "https://example.com/deepseek-v4",
                "snippet": "短摘要",
                "summary": "长摘要内容",
                "content": "正文内容",
                "site_name": "Example News",
                "site_icon": None,
                "date": "2026-06-01T00:00:00+00:00",
                "authority": 3,
            },
            {
                "rerank_score": 0.8,
                "title": "低质结果",
                "url": "https://example.com/low",
                "snippet": "",
                "summary": "",
                "content": "",
                "site_name": None,
                "site_icon": None,
                "date": None,
                "authority": -1,
            },
        ]
    }
    text = format_search_results(search_response, query="DeepSeek V4", max_summary_chars=100, max_content_chars=50)
    check("搜索结果包含标题", "DeepSeek V4 模型发布" in text)
    check("搜索结果包含链接", "https://example.com/deepseek-v4" in text)
    check("搜索结果包含权威性", "权威性: 3/4" in text)
    check("搜索结果包含相关分", "相关分: 2.40" in text)
    check("无日期显示未知", "发布时间: 未知" in text)
    check("空结果有明确提示", "没有找到任何结果" in format_search_results({"results": []}, query="x"))

    long_content = "x" * 200
    truncated = format_search_results(
        {"results": [{"title": "t", "url": "u", "summary": "s", "content": long_content, "authority": 0, "rerank_score": 1.0}]},
        query="q", max_content_chars=50,
    )
    check("正文超限被截断", "已截断" in truncated and "x" * 51 not in truncated)

    fetch_text = format_fetch_result({"title": "页面标题", "content": "# 正文"}, url="https://example.com/a")
    check("抓取结果包含标题", "页面标题" in fetch_text)
    check("抓取结果包含正文", "# 正文" in fetch_text)
    empty_fetch = format_fetch_result({"title": "", "content": ""}, url="https://example.com/b")
    check("空正文有明确提示", "没有可提取的正文内容" in empty_fetch)


def test_client_error_and_retry() -> None:
    """验证客户端错误解析与重试逻辑（mock aiohttp 会话）。"""

    print("== 客户端错误与重试 ==")
    from deepseek_client import DeepSeekAPIError, DeepSeekClient, _extract_error_message

    payload = {"error": {"message": "'count' must be between 1 and 50", "type": "invalid_request_error", "code": "invalid_request_error", "param": None}}
    check("错误信息提取", _extract_error_message(payload, status=400) == "'count' must be between 1 and 50")

    error = DeepSeekAPIError("boom", status=429)
    check("429 标记为可重试", error.retryable)
    check("400 标记为不可重试", not DeepSeekAPIError("bad", status=400).retryable)
    check("网络错误（无状态码）标记为可重试", DeepSeekAPIError("net").retryable)

    async def scenario() -> None:
        client = DeepSeekClient(api_key="sk-test", max_retries=2, min_request_interval_seconds=0)

        # 场景 1：400 不重试，直接抛出
        call_count = {"n": 0}

        class Resp400:
            """模拟 400 响应（异步上下文管理器）。"""

            status = 400

            async def json(self, content_type: Any = None) -> Any:
                return payload

            async def text(self) -> str:
                return ""

            async def __aenter__(self) -> "Resp400":
                return self

            async def __aexit__(self, *args: Any) -> bool:
                return False

        class Session400:
            """模拟返回 400 的会话。"""

            closed = False

            def post(self, *args: Any, **kwargs: Any) -> Resp400:
                call_count["n"] += 1
                return Resp400()

            async def close(self) -> None:
                self.closed = True

        client._session = Session400()  # type: ignore[assignment]
        try:
            await client.search(query="q")
            check("400 抛出异常", False)
        except DeepSeekAPIError as exc:
            check("400 抛出异常且不重试", exc.status == 400 and call_count["n"] == 1, f"调用 {call_count['n']} 次")
        await client.close()

        # 场景 2：429 重试两次后成功
        attempt = {"n": 0}

        class Resp429:
            status = 429

            async def json(self, content_type: Any = None) -> Any:
                return {"error": {"message": "rate limited", "type": "rate_limit_error", "code": "rate_limit_error"}}

            async def text(self) -> str:
                return ""

            async def __aenter__(self) -> "Resp429":
                return self

            async def __aexit__(self, *args: Any) -> bool:
                return False

        class Resp200:
            status = 200

            async def json(self, content_type: Any = None) -> Any:
                return {"results": []}

            async def text(self) -> str:
                return ""

            async def __aenter__(self) -> "Resp200":
                return self

            async def __aexit__(self, *args: Any) -> bool:
                return False

        class SessionRetry:
            closed = False

            def post(self, *args: Any, **kwargs: Any) -> Any:
                attempt["n"] += 1
                if attempt["n"] < 3:
                    return Resp429()
                return Resp200()

            async def close(self) -> None:
                self.closed = True

        client2 = DeepSeekClient(api_key="sk-test", max_retries=2, min_request_interval_seconds=0)
        client2._session = SessionRetry()  # type: ignore[assignment]
        result = await client2.search(query="q")
        check("429 重试后成功", attempt["n"] == 3 and result == {"results": []}, f"调用 {attempt['n']} 次")
        await client2.close()

        # 场景 3：请求头包含 Bearer 鉴权、自定义 URL、请求体字段
        captured: Dict[str, Any] = {}

        class RespCapture:
            status = 200

            async def json(self, content_type: Any = None) -> Any:
                return {"results": []}

            async def text(self) -> str:
                return ""

            async def __aenter__(self) -> "RespCapture":
                return self

            async def __aexit__(self, *args: Any) -> bool:
                return False

        class SessionCapture:
            closed = False

            def post(self, url: str, json: Any = None, headers: Any = None, **kwargs: Any) -> RespCapture:
                captured["url"] = url
                captured["headers"] = headers
                captured["payload"] = json
                return RespCapture()

            async def close(self) -> None:
                self.closed = True

        client3 = DeepSeekClient(api_key="sk-abc", search_url="https://proxy.example.com/v1/tools/search", min_request_interval_seconds=0)
        client3._session = SessionCapture()  # type: ignore[assignment]
        await client3.search(query="测试", count=5, ori_question="用户问题", need_content=True, localization="1.2.3.4")
        check("使用自定义 search_url", captured["url"] == "https://proxy.example.com/v1/tools/search")
        check("Bearer 鉴权头", captured["headers"].get("Authorization") == "Bearer sk-abc")
        check("请求体字段完整",
              captured["payload"] == {"query": "测试", "count": 5, "need_content": True, "ori_question": "用户问题", "localization": "1.2.3.4"},
              f"实际 {captured['payload']}")
        await client3.close()

        # 场景 4：节流生效
        client4 = DeepSeekClient(api_key="sk-test", min_request_interval_seconds=0.1)
        started = asyncio.get_event_loop().time()
        await client4._throttle()
        await client4._throttle()
        elapsed = asyncio.get_event_loop().time() - started
        check("节流间隔生效", elapsed >= 0.09, f"实际 {elapsed:.3f}s")
        await client4.close()

    asyncio.run(scenario())


def test_hook_enhanced_mode() -> None:
    """验证增强模式 Hook 注入逻辑。"""

    print("== 增强模式 Hook ==")
    instance = load_plugin()
    # 注入测试用 API Key 并重建客户端（无 Key 时 Hook 按设计不注入）
    instance.config.api.api_key = "sk-test"
    instance._client = instance._build_client()

    tool_definitions = [
        {"type": "function", "function": {"name": "deepseek_web_search", "description": "基础搜索描述", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "deepseek_web_fetch", "description": "基础抓取描述", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "other_tool", "description": "别的工具", "parameters": {"type": "object", "properties": {}}}},
    ]
    base_items = [
        {"item_type": "SystemMessageItem", "meta": {"item_id": "sys-1", "logical_turn_id": None, "timestamp": "2026-10-18T00:00:00+00:00"}, "parts": [{"type": "text", "text": "系统提示"}]},
        {"item_type": "UserMessageItem", "meta": {"item_id": "usr-1", "logical_turn_id": "turn-1", "timestamp": "2026-10-18T00:00:01+00:00"}, "parts": [{"type": "text", "text": "今天有什么新闻"}]},
    ]

    async def scenario() -> None:
        # 普通模式：不注入
        result = await instance.push_enhanced_mode(
            items=list(base_items),
            item_schema_version=1,
            tool_definitions=[copy.deepcopy(d) for d in tool_definitions],
            selected_history_count=1,
            built_message_count=2,
            selection_reason="test",
            session_id="s-1",
        )
        check("普通模式不修改 kwargs", "modified_kwargs" not in result)

        # 增强模式：注入指令 + 强化描述
        instance._mode_override = "enhanced"
        result = await instance.push_enhanced_mode(
            items=[copy.deepcopy(d) for d in base_items],
            item_schema_version=1,
            tool_definitions=[copy.deepcopy(d) for d in tool_definitions],
            selected_history_count=1,
            built_message_count=2,
            selection_reason="test",
            session_id="s-1",
        )
        check("增强模式返回 modified_kwargs", "modified_kwargs" in result)
        modified_kwargs = result.get("modified_kwargs", {})
        new_items = modified_kwargs.get("items", [])
        check("注入后 items 增加 1 条", len(new_items) == len(base_items) + 1)
        injected = new_items[-1] if new_items else {}
        check("注入的是 SystemMessageItem", injected.get("item_type") == "SystemMessageItem")
        check("注入消息 item_id 固定", injected.get("meta", {}).get("item_id") == "deepseek_search_enhanced_reminder")
        check("注入消息包含督促文本", "增强模式" in str(injected.get("parts")))
        new_definitions = modified_kwargs.get("tool_definitions", [])
        search_def = next(d for d in new_definitions if d["function"]["name"] == "deepseek_web_search")
        fetch_def = next(d for d in new_definitions if d["function"]["name"] == "deepseek_web_fetch")
        other_def = next(d for d in new_definitions if d["function"]["name"] == "other_tool")
        check("搜索工具描述被强化", "增强模式" in search_def["function"]["description"])
        check("抓取工具描述被强化", "增强模式" in fetch_def["function"]["description"])
        check("其他工具描述不受影响", other_def["function"]["description"] == "别的工具")
        check("原始描述保留", search_def["function"]["description"].startswith("基础搜索描述"))
        check("其它 kwargs 原样带回", modified_kwargs.get("session_id") == "s-1" and modified_kwargs.get("item_schema_version") == 1)

        # 重复注入防护：items 中已有督促消息时只强化描述
        items_with_reminder = [*base_items, injected]
        result2 = await instance.push_enhanced_mode(
            items=[dict(d) for d in items_with_reminder],
            item_schema_version=1,
            tool_definitions=[copy.deepcopy(d) for d in tool_definitions],
            session_id="s-1",
        )
        check("已有督促消息时不重复注入", len(result2["modified_kwargs"]["items"]) == len(items_with_reminder))

        # 子代理场景：工具列表中无本插件工具时不注入
        result3 = await instance.push_enhanced_mode(
            items=[copy.deepcopy(d) for d in base_items],
            item_schema_version=1,
            tool_definitions=[{"type": "function", "function": {"name": "other_tool", "description": "x", "parameters": {}}}],
            session_id="s-1",
        )
        check("无本插件工具时不注入", "modified_kwargs" not in result3)

        # 空工具列表不注入
        result4 = await instance.push_enhanced_mode(items=list(base_items), tool_definitions=[], session_id="s-1")
        check("空工具列表不注入", "modified_kwargs" not in result4)

        instance._mode_override = None

    asyncio.run(scenario())


def test_command() -> None:
    """验证命令处理逻辑。"""

    print("== 命令处理 ==")
    instance = load_plugin()

    async def scenario() -> None:
        # /dssearch mode enhanced
        success, reply, intercept = await instance.handle_dssearch_command(
            stream_id="s-1", text="/dssearch mode enhanced", matched_groups={"action": "mode", "mode": "enhanced"},
        )
        check("切换 enhanced 成功", success and instance.current_mode == "enhanced")
        check("命令返回拦截", intercept == 1)
        check("命令有回复文本", "已切换运行模式" in reply)
        check("消息已发送", len(instance.ctx.send.sent) >= 1)

        # 普通模式不注入（已在 hook 测试覆盖），这里验证状态文本
        success, reply, _ = await instance.handle_dssearch_command(
            stream_id="s-1", text="/dssearch status", matched_groups={"action": "status", "mode": None},
        )
        check("状态文本包含增强模式", "运行模式: enhanced" in reply)
        check("状态文本包含接口地址", "api.deepseek.com/v1/tools/search" in reply)

        # 切回普通
        await instance.handle_dssearch_command(
            stream_id="s-1", text="/dssearch mode normal", matched_groups={"action": "mode", "mode": "normal"},
        )
        check("切回 normal 成功", instance.current_mode == "normal")

        # 无参数 → 状态
        _, reply, _ = await instance.handle_dssearch_command(stream_id="s-1", text="/dssearch", matched_groups={})
        check("无参数显示状态", "DeepSeek 联网搜索插件状态" in reply)

        # 文本回退解析（别名匹配场景 matched_groups 为空）
        _, reply, _ = await instance.handle_dssearch_command(stream_id="s-1", text="/dssearch mode enhanced", matched_groups={})
        check("文本回退解析生效", instance.current_mode == "enhanced")
        await instance.handle_dssearch_command(stream_id="s-1", text="/dssearch mode normal", matched_groups={})

        # help
        _, reply, _ = await instance.handle_dssearch_command(stream_id="s-1", text="/dssearch help", matched_groups={"action": "help", "mode": None})
        check("帮助文本", "/dssearch mode enhanced" in reply)

    asyncio.run(scenario())


def test_command_pattern() -> None:
    """验证命令正则的匹配行为。"""

    print("== 命令正则 ==")
    import re

    pattern = re.compile(plugin_module.COMMAND_PATTERN)
    cases = {
        "/dssearch": {"action": None, "mode": None},
        "/dssearch status": {"action": "status", "mode": None},
        "/dssearch mode": {"action": "mode", "mode": None},
        "/dssearch mode enhanced": {"action": "mode", "mode": "enhanced"},
        "/dssearch mode normal": {"action": "mode", "mode": "normal"},
        "/dssearch enhanced": {"action": None, "mode": "enhanced"},
        "/dssearch help": {"action": "help", "mode": None},
        "/dssearch   mode   enhanced": {"action": "mode", "mode": "enhanced"},
    }
    for text, expected in cases.items():
        match = pattern.search(text)
        ok = match is not None and match.groupdict() == expected
        check(f"pattern 匹配 {text!r}", ok, f"实际 {match.groupdict() if match else None}")

    non_matches = ["/dssearch mode super", "/dssearchx", "hello /dssearch mode enhanced", "/dssearch mode enhanced now"]
    for text in non_matches:
        check(f"pattern 不匹配 {text!r}", pattern.search(text) is None)


def test_tools_with_mock_client() -> None:
    """验证工具处理器（mock 客户端）。"""

    print("== 工具处理器 ==")
    instance = load_plugin()
    instance._client = instance._build_client()

    async def scenario() -> None:
        # 未配置 API Key
        empty_instance = load_plugin()
        result = await empty_instance.handle_web_search(query="测试")
        check("无 API Key 返回失败", result["success"] is False and "API Key" in result["error"])
        result = await empty_instance.handle_web_fetch(url="https://example.com")
        check("抓取无 API Key 返回失败", result["success"] is False and "API Key" in result["error"])

        # mock 客户端
        mock_client = MagicMock()
        mock_client.configured = True
        mock_client.search = AsyncMock(return_value={"results": [{"title": "标题", "url": "https://a.com", "summary": "摘要", "authority": 3, "rerank_score": 2.0, "date": None, "site_name": "站点"}]})
        mock_client.fetch = AsyncMock(return_value={"title": "页面", "content": "正文"})
        instance._client = mock_client

        result = await instance.handle_web_search(query="测试", count=3, need_content=True, ori_question="问题")
        check("搜索成功", result["success"] is True and "标题" in result["content"])
        call_kwargs = mock_client.search.call_args.kwargs
        check("搜索参数传递正确", call_kwargs["query"] == "测试" and call_kwargs["count"] == 3 and call_kwargs["need_content"] is True and call_kwargs["ori_question"] == "问题")

        # count 越界收敛
        await instance.handle_web_search(query="测试", count=999)
        check("count 上限收敛到配置 max_count", mock_client.search.call_args.kwargs["count"] == 20)
        await instance.handle_web_search(query="测试", count=0)
        check("count 下限收敛到 1", mock_client.search.call_args.kwargs["count"] == 1)
        await instance.handle_web_search(query="测试", count="abc")
        check("count 非法回退默认值", mock_client.search.call_args.kwargs["count"] == 10)

        # 空 query
        result = await instance.handle_web_search(query="   ")
        check("空 query 返回失败", result["success"] is False)

        # 抓取
        result = await instance.handle_web_fetch(url="https://example.com/article")
        check("抓取成功", result["success"] is True and "页面" in result["content"])

        # URL 校验
        for bad_url in ["", "ftp://example.com", "https://user:pass@example.com", "x" * 9000]:
            result = await instance.handle_web_fetch(url=bad_url)
            check(f"非法 url 被拒绝: {bad_url[:30]!r}", result["success"] is False)

        # API 错误传播
        from deepseek_client import DeepSeekAPIError
        mock_client.search = AsyncMock(side_effect=DeepSeekAPIError("rate limited", status=429))
        result = await instance.handle_web_search(query="测试")
        check("API 错误转为失败结果", result["success"] is False and "搜索失败" in result["error"])

    asyncio.run(scenario())


def test_lifecycle() -> None:
    """验证生命周期回调。"""

    print("== 生命周期 ==")
    instance = load_plugin()

    async def scenario() -> None:
        await instance.on_load()
        check("on_load 后客户端就绪", instance._client is not None)
        # 配置热更新：切换模式并重建客户端
        instance.config.mode.mode = "enhanced"
        await instance.on_config_update("self", instance.get_plugin_config_data(), "1.0.0")
        check("配置热更新后模式来自配置", instance.current_mode == "enhanced")
        await instance.on_config_update("bot", {}, "1.0.0")
        check("非 self 范围被忽略", instance.current_mode == "enhanced")
        instance._mode_override = "normal"
        await instance.on_config_update("self", instance.get_plugin_config_data(), "1.0.0")
        check("配置热更新清除命令级覆盖", instance.current_mode == "enhanced")
        await instance.on_unload()
        check("on_unload 后客户端释放", instance._client is None)

    asyncio.run(scenario())


def main() -> None:
    """运行全部验证。"""

    test_components()
    test_config_model()
    test_client_formatting()
    test_client_error_and_retry()
    test_hook_enhanced_mode()
    test_command()
    test_command_pattern()
    test_tools_with_mock_client()
    test_lifecycle()
    print(f"\n结果: {PASS} 通过, {FAIL} 失败")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
