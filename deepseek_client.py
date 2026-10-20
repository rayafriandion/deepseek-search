"""DeepSeek Search / Fetch API 客户端。

封装 DeepSeek 官方网页搜索（POST /v1/tools/search）与网页抓取
（POST /v1/tools/fetch）接口，包括 Bearer 鉴权、请求节流、可重试错误的
指数退避重试、OpenAI 兼容错误格式解析，以及面向 LLM 的结果文本格式化。

接口文档：
- https://api-docs.deepseek.com/zh-cn/api/search
- https://api-docs.deepseek.com/zh-cn/api/fetch
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Dict, List, Optional

import aiohttp

# DeepSeek 官方默认接口地址
DEFAULT_SEARCH_URL = "https://api.deepseek.com/v1/tools/search"
DEFAULT_FETCH_URL = "https://api.deepseek.com/v1/tools/fetch"

# 按官方指南，遇到这些状态码应等待后重试；其它 4xx 多为请求问题，重试无意义
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

# DeepSeek 账号级 QPS 限制为 5，节流间隔默认 0.25 秒（约 4 QPS）留出余量
DEFAULT_MIN_REQUEST_INTERVAL_SECONDS = 0.25

# 单次请求默认超时（秒）
DEFAULT_TIMEOUT_SECONDS = 30.0

# 重试基础退避时间（秒），按 attempt 指数增长并叠加抖动
RETRY_BASE_BACKOFF_SECONDS = 0.5


class DeepSeekAPIError(Exception):
    """DeepSeek API 调用失败。

    Attributes:
        status: HTTP 状态码；网络层失败时为 None。
        code: DeepSeek 返回的错误码（如 invalid_request_error）。
        error_type: DeepSeek 返回的错误类型。
    """

    def __init__(
        self,
        message: str,
        *,
        status: Optional[int] = None,
        code: str = "",
        error_type: str = "",
    ) -> None:
        """初始化 API 错误。

        Args:
            message: 面向调用方的错误描述。
            status: HTTP 状态码；网络层失败时为 None。
            code: DeepSeek 返回的错误码。
            error_type: DeepSeek 返回的错误类型。
        """

        super().__init__(message)
        self.status = status
        self.code = code
        self.error_type = error_type

    @property
    def retryable(self) -> bool:
        """判断该错误是否值得等待后重试。"""

        return self.status is None or self.status in RETRYABLE_STATUS_CODES


def _extract_error_message(payload: Any, *, status: Optional[int]) -> str:
    """从 DeepSeek 错误响应中提取可读信息。

    DeepSeek 使用 OpenAI 兼容错误格式：
    ``{"error": {"message": ..., "type": ..., "code": ..., "param": null}}``。

    Args:
        payload: 解析后的响应体。
        status: HTTP 状态码。

    Returns:
        str: 适合展示给模型或用户的错误描述。
    """

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "").strip()
            if message:
                return message
        message = str(payload.get("message") or "").strip()
        if message:
            return message
    text = str(payload or "").strip()
    if text:
        return text[:500]
    if status is not None:
        return f"HTTP {status}"
    return "未知错误"


class DeepSeekClient:
    """DeepSeek Search / Fetch 异步客户端。

    客户端内部复用同一个 aiohttp 会话，并按 min_request_interval 做请求节流，
    避免触发账号级 QPS 限制（Search / Fetch 各 5 QPS）。
    """

    def __init__(
        self,
        *,
        api_key: str,
        search_url: str = DEFAULT_SEARCH_URL,
        fetch_url: str = DEFAULT_FETCH_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = 2,
        min_request_interval_seconds: float = DEFAULT_MIN_REQUEST_INTERVAL_SECONDS,
    ) -> None:
        """初始化客户端。

        Args:
            api_key: DeepSeek API Key（Bearer 鉴权）。
            search_url: Search 接口完整地址。
            fetch_url: Fetch 接口完整地址。
            timeout_seconds: 单次请求超时时间（秒）。
            max_retries: 可重试错误的最大重试次数。
            min_request_interval_seconds: 两次请求之间的最小间隔（秒）。
        """

        self.api_key = api_key.strip()
        self.search_url = search_url.strip() or DEFAULT_SEARCH_URL
        self.fetch_url = fetch_url.strip() or DEFAULT_FETCH_URL
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.max_retries = max(0, int(max_retries))
        self.min_request_interval_seconds = max(0.0, float(min_request_interval_seconds))
        self._session: Optional[aiohttp.ClientSession] = None
        self._throttle_lock = asyncio.Lock()
        self._last_request_monotonic = 0.0

    @property
    def configured(self) -> bool:
        """判断 API Key 是否已配置。"""

        return bool(self.api_key)

    async def close(self) -> None:
        """关闭底层 HTTP 会话。"""

        session = self._session
        self._session = None
        if session is not None and not session.closed:
            await session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        """按需创建并复用 HTTP 会话。

        Returns:
            aiohttp.ClientSession: 可用的 HTTP 会话。
        """

        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _throttle(self) -> None:
        """保证相邻请求间隔不低于 min_request_interval，避免触发 QPS 限制。"""

        async with self._throttle_lock:
            now = time.monotonic()
            wait_seconds = self._last_request_monotonic + self.min_request_interval_seconds - now
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
            self._last_request_monotonic = time.monotonic()

    async def _post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """执行一次带节流与重试的 POST 请求。

        Args:
            url: 目标接口地址。
            payload: JSON 请求体。

        Returns:
            Dict[str, Any]: 解析后的响应 JSON。

        Raises:
            DeepSeekAPIError: 重试耗尽后仍然失败，或遇到不应重试的错误。
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Optional[DeepSeekAPIError] = None

        for attempt in range(self.max_retries + 1):
            await self._throttle()
            try:
                session = await self._get_session()
                async with session.post(url, json=payload, headers=headers) as response:
                    status = response.status
                    # 响应体解析失败时退化为原始文本，交给错误提取逻辑处理
                    try:
                        body: Any = await response.json(content_type=None)
                    except (aiohttp.ContentTypeError, ValueError):
                        body = await response.text()
                    if status == 200:
                        if isinstance(body, dict):
                            return body
                        raise DeepSeekAPIError(
                            f"DeepSeek 接口返回了非对象响应: {str(body)[:200]}",
                            status=status,
                        )
                    error = DeepSeekAPIError(
                        _extract_error_message(body, status=status),
                        status=status,
                        code=_extract_error_field(body, "code"),
                        error_type=_extract_error_field(body, "type"),
                    )
            except DeepSeekAPIError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # 网络层错误（连接失败、超时等）按可重试处理
                error = DeepSeekAPIError(
                    f"网络错误: {exc.__class__.__name__}: {exc}",
                    status=None,
                )

            last_error = error
            if not error.retryable or attempt >= self.max_retries:
                break
            # 指数退避 + 抖动，避免与其它调用方同步重试
            backoff = RETRY_BASE_BACKOFF_SECONDS * (2**attempt) + random.uniform(0, 0.25)
            await asyncio.sleep(backoff)

        assert last_error is not None
        raise last_error

    async def search(
        self,
        *,
        query: str,
        count: int = 10,
        ori_question: Optional[str] = None,
        need_content: bool = False,
        localization: str = "",
    ) -> Dict[str, Any]:
        """调用网页搜索接口。

        Args:
            query: 搜索词，1-128 个字符，支持 site:、-site:、引号等语法。
            count: 返回条数，1-50。
            ori_question: 触发搜索的用户原始问题，便于理解搜索意图。
            need_content: 是否在 content 字段返回网页正文。
            localization: 用于优先返回本地化结果的 IP 地址。

        Returns:
            Dict[str, Any]: 含 results 数组的原始响应。

        Raises:
            DeepSeekAPIError: 接口调用失败。
        """

        payload: Dict[str, Any] = {
            "query": query,
            "count": count,
            "need_content": need_content,
        }
        if ori_question:
            payload["ori_question"] = ori_question
        if localization:
            payload["localization"] = localization
        return await self._post(self.search_url, payload)

    async def fetch(self, *, url: str) -> Dict[str, Any]:
        """调用网页抓取接口。

        Args:
            url: 目标网页地址，须为含 host 的 HTTP/HTTPS URL。

        Returns:
            Dict[str, Any]: 含 title 与 content（Markdown）的原始响应。

        Raises:
            DeepSeekAPIError: 接口调用失败。
        """

        return await self._post(self.fetch_url, {"url": url})


def _extract_error_field(payload: Any, field: str) -> str:
    """从 DeepSeek 错误响应中提取指定字段。

    Args:
        payload: 解析后的响应体。
        field: 目标字段名（如 code、type）。

    Returns:
        str: 字段值；缺失或结构不符时返回空字符串。
    """

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            value = error.get(field)
            if isinstance(value, str):
                return value.strip()
    return ""


def _truncate(text: str, max_chars: int) -> str:
    """按字符数截断文本，超限时追加省略标记。

    Args:
        text: 原始文本。
        max_chars: 最大字符数；小于等于 0 表示不截断。

    Returns:
        str: 截断后的文本。
    """

    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "……[已截断]"


def _format_optional(value: Any, default: str = "未知") -> str:
    """将可能为 None 的字段格式化为字符串。

    Args:
        value: 原始字段值。
        default: 值为空时的占位文本。

    Returns:
        str: 格式化后的文本。
    """

    if value is None or value == "":
        return default
    return str(value)


def format_search_results(
    response: Dict[str, Any],
    *,
    query: str = "",
    max_summary_chars: int = 1200,
    max_content_chars: int = 4000,
) -> str:
    """将搜索接口响应格式化为面向 LLM 的文本。

    Args:
        response: 搜索接口原始响应（含 results 数组）。
        query: 本次搜索使用的查询词，用于回显。
        max_summary_chars: 单条摘要最大字符数。
        max_content_chars: 单条网页正文最大字符数。

    Returns:
        str: 结构化搜索结果文本；无结果时给出明确提示。
    """

    raw_results = response.get("results")
    results: List[Dict[str, Any]] = [item for item in raw_results if isinstance(item, dict)] if isinstance(raw_results, list) else []

    if not results:
        return f"搜索「{query}」没有找到任何结果。可以尝试更换关键词、使用 site: 限定站点或调整搜索语法后重试。"

    lines: List[str] = [f"搜索「{query}」共返回 {len(results)} 条结果："]
    for index, item in enumerate(results, start=1):
        title = _format_optional(item.get("title"), "无标题")
        url = _format_optional(item.get("url"), "无链接")
        site_name = _format_optional(item.get("site_name"), "未知站点")
        date = _format_optional(item.get("date"), "未知时间")
        authority = item.get("authority")
        authority_text = f"{authority}/4" if isinstance(authority, int) else "未知"
        rerank_score = item.get("rerank_score")
        score_text = f"{rerank_score:.2f}" if isinstance(rerank_score, (int, float)) else "未知"
        summary = _truncate(str(item.get("summary") or "").strip(), max_summary_chars) or "无摘要"
        block = [
            f"[{index}] {title}",
            f"链接: {url}",
            f"来源: {site_name} | 发布时间: {date} | 权威性: {authority_text} | 相关分: {score_text}",
            f"摘要: {summary}",
        ]
        content = str(item.get("content") or "").strip()
        if content:
            block.append(f"正文: {_truncate(content, max_content_chars)}")
        lines.append("\n".join(block))
    lines.append("提示: 权威性 3 及以上为极权威结果，相关分 1.5 及以上为优质结果；如需全文可用 deepseek_web_fetch 抓取链接。")
    return "\n\n".join(lines)


def format_fetch_result(
    response: Dict[str, Any],
    *,
    url: str = "",
    max_content_chars: int = 12000,
) -> str:
    """将抓取接口响应格式化为面向 LLM 的文本。

    Args:
        response: 抓取接口原始响应（含 title 与 content）。
        url: 本次抓取的目标地址，用于回显。
        max_content_chars: 正文最大字符数。

    Returns:
        str: 结构化抓取结果文本。
    """

    title = str(response.get("title") or "").strip() or "无标题"
    content = str(response.get("content") or "").strip()
    if not content:
        return f"抓取 {url} 成功，但页面没有可提取的正文内容（可能为空白页、纯脚本页面或内容类型不受支持）。"
    body = _truncate(content, max_content_chars)
    return f"页面标题: {title}\n来源: {url}\n正文（Markdown）:\n{body}"
