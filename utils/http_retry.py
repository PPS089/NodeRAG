"""
HTTP 重试工具 — 通过 monkey-patch requests.Session.send 实现全局重试。

用法（零侵入，原代码无需修改）：

    from utils.http_retry import install_retry
    install_retry()

此后所有 requests.post / get / put 调用在网络抖动、限流、5xx 时自动重试。

环境变量：
    HTTP_RETRY_ENABLED=1        启用重试（默认启用）
    HTTP_RETRY_MAX=3            最大重试次数（默认 3）
    HTTP_RETRY_BACKOFF=1.0      基础退避秒数（默认 1.0），实际延迟 = backoff * 2^attempt
    HTTP_RETRY_MAX_DELAY=32     最大延迟秒数上限（默认 32）
    HTTP_RETRY_LOG=1            是否打印重试日志到 stderr（默认启用）
"""
from __future__ import annotations

import os
import random
import sys
import threading
import time
from typing import Optional, Set, Tuple, Type

import requests
import urllib3.exceptions


# ---------------------------------------------------------------------------
# 环境变量配置
# ---------------------------------------------------------------------------

def _env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# 可重试 vs 不可重试 错误分类
# ---------------------------------------------------------------------------

# HTTP 状态码：429（限流）和 5xx（服务端错误）可重试
RETRYABLE_STATUS_CODES: Set[int] = {429}
RETRYABLE_STATUS_RANGES: Tuple[int, int] = (500, 600)

# 网络层异常：连接失败 / 超时可重试
RETRYABLE_EXCEPTIONS: Tuple[Type[BaseException], ...] = (
    requests.ConnectionError,
    requests.Timeout,
    urllib3.exceptions.TimeoutError,
    urllib3.exceptions.ConnectTimeoutError,
    urllib3.exceptions.ReadTimeoutError,
    urllib3.exceptions.NewConnectionError,
    urllib3.exceptions.MaxRetryError,
)


def is_retryable(response_or_error: requests.Response | BaseException) -> bool:
    """判断一个响应或异常是否应该重试。"""
    if isinstance(response_or_error, requests.Response):
        status = response_or_error.status_code
        if status in RETRYABLE_STATUS_CODES:
            return True
        if RETRYABLE_STATUS_RANGES[0] <= status < RETRYABLE_STATUS_RANGES[1]:
            return True
        return False

    if isinstance(response_or_error, RETRYABLE_EXCEPTIONS):
        return True

    # 某些异常包裹了底层 urllib3 错误
    cause = getattr(response_or_error, "__cause__", None)
    while cause is not None:
        if isinstance(cause, RETRYABLE_EXCEPTIONS):
            return True
        cause = getattr(cause, "__cause__", None)

    return False


def describe_error(response_or_error: requests.Response | BaseException) -> str:
    """生成可读的错误描述，用于日志。"""
    if isinstance(response_or_error, requests.Response):
        return f"HTTP {response_or_error.status_code}"
    return f"{type(response_or_error).__name__}: {response_or_error}"


# ---------------------------------------------------------------------------
# 指数退避 + 抖动
# ---------------------------------------------------------------------------

def backoff_delay(attempt: int, backoff_factor: float, max_delay: float) -> float:
    """计算第 attempt 次重试的等待秒数（含 ±25% 随机抖动）。"""
    base = min(backoff_factor * (2 ** attempt), max_delay)
    jitter = base * 0.25
    return base + random.uniform(-jitter, jitter)


# ---------------------------------------------------------------------------
# Monkey-patch requests.Session.send
# ---------------------------------------------------------------------------

_original_send = None  # 保存原始方法
_lock = threading.Lock()
_installed = False


def install_retry() -> None:
    """安装 HTTP 重试 monkey-patch（幂等，重复调用不会重复安装）。"""
    global _original_send, _installed

    with _lock:
        if _installed:
            return

        if not _env_bool("HTTP_RETRY_ENABLED", default=True):
            _installed = True
            return

        _original_send = requests.Session.send
        max_retries = _env_int("HTTP_RETRY_MAX", 3)
        backoff_factor = _env_float("HTTP_RETRY_BACKOFF", 1.0)
        max_delay = _env_float("HTTP_RETRY_MAX_DELAY", 32.0)
        log_enabled = _env_bool("HTTP_RETRY_LOG", default=True)

        def _send_with_retry(self, request, **kwargs):
            # stream=True 时不重试（响应体已部分发送给调用方）
            if kwargs.get("stream", False):
                return _original_send(self, request, **kwargs)

            last_error: requests.Response | BaseException | None = None

            for attempt in range(max_retries + 1):
                try:
                    response = _original_send(self, request, **kwargs)

                    if is_retryable(response) and attempt < max_retries:
                        last_error = response
                        delay = backoff_delay(attempt, backoff_factor, max_delay)
                        if log_enabled:
                            print(
                                f"[http_retry] {describe_error(response)} → "
                                f"第 {attempt + 1}/{max_retries} 次重试，等待 {delay:.1f}s  "
                                f"URL: {request.url}",
                                file=sys.stderr,
                            )
                        time.sleep(delay)
                        continue

                    return response

                except RETRYABLE_EXCEPTIONS as exc:
                    last_error = exc
                    if attempt < max_retries:
                        delay = backoff_delay(attempt, backoff_factor, max_delay)
                        if log_enabled:
                            print(
                                f"[http_retry] {describe_error(exc)} → "
                                f"第 {attempt + 1}/{max_retries} 次重试，等待 {delay:.1f}s  "
                                f"URL: {request.url}",
                                file=sys.stderr,
                            )
                        time.sleep(delay)
                        continue
                    raise

                # 不可重试的异常直接抛出
                except Exception:
                    raise

            # 所有重试已用尽
            if isinstance(last_error, requests.Response):
                last_error.raise_for_status()
            if isinstance(last_error, BaseException):
                raise last_error
            # 理论上不会走到这里
            raise RuntimeError("HTTP 重试耗尽但无错误信息")

        requests.Session.send = _send_with_retry
        _installed = True


def uninstall_retry() -> None:
    """卸载 HTTP 重试 monkey-patch，恢复原始 requests.Session.send。"""
    global _installed

    with _lock:
        if not _installed:
            return
        if _original_send is not None:
            requests.Session.send = _original_send
        _installed = False


def is_retry_installed() -> bool:
    """返回当前是否已安装重试。"""
    return _installed
