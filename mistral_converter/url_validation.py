"""SSRF-safe document URL validation and signed-URL error classification."""

import ipaddress
import socket
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as DnsTimeoutError
from typing import Any, Optional, Tuple

import config
import utils

from .facade import attr

logger = utils.logger
_SIGNED_URL_SPECIFIC_HINTS: Tuple[str, ...] = (
    "signed url",
    "url has expired",
    "url expired",
    "signature expired",
    "signature mismatch",
    "expired signature",
)

# Fetch/download failures are sufficient alone (object store could not serve the URL).
_SIGNED_URL_FETCH_HINTS: Tuple[str, ...] = (
    "failed to fetch document",
    "could not download",
)

# Note: bare "403 forbidden" / "access denied" are intentionally NOT always-true
# hints. Those only become retryable when the message also contains a
# signed-URL-specific hint above (e.g. "403 Forbidden: signed URL expired").

# Tokens that mean the failure is permanent (API key / account) and retrying
# with a new URL will not help.
_PERMANENT_AUTH_HINTS: Tuple[str, ...] = (
    "401",
    "unauthorized",
    "api key",
    "invalid api key",
    "authentication failed",
)


def is_signed_url_expiry_error(message: Optional[str]) -> bool:
    """Classify a QnA/OCR error message as a likely signed-URL expiry.

    Returns True only for messages that look like the object store rejected
    the signed URL (e.g. "signed URL has expired", fetch/download failures).
    Bare 403 / access-denied messages are retryable only when also paired with
    a signed-URL-specific hint. Returns False for Mistral API auth failures.
    """
    if not message:
        return False
    lowered = message.lower()
    if any(hint in lowered for hint in _PERMANENT_AUTH_HINTS):
        return False
    if any(hint in lowered for hint in _SIGNED_URL_FETCH_HINTS):
        return True
    has_signed_url_hint = any(hint in lowered for hint in _SIGNED_URL_SPECIFIC_HINTS)
    if has_signed_url_hint:
        return True
    # Bare 403 / access-denied is not enough without a signed-URL-specific hint.
    return False


# ============================================================================
# OCR Quality Assessment
# ============================================================================
# NOTE: All OCR quality thresholds are now configured via config.py
# This ensures .env settings are honored as documented in README.md
# See: config.OCR_MIN_TEXT_LENGTH, config.OCR_MIN_UNIQUENESS_RATIO, etc.

# Process-global page counter — suitable for CLI use.  A multi-tenant
# service would need per-request counters instead.
_dns_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mistral-dns")


def _is_forbidden_address(addr: Any) -> bool:
    """Return True if *addr* points to a private/internal/reserved network."""
    if addr.is_private or addr.is_reserved or addr.is_loopback or addr.is_link_local or addr.is_multicast:
        return True
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        mapped = addr.ipv4_mapped
        if mapped.is_private or mapped.is_loopback or mapped.is_reserved or mapped.is_link_local:
            return True
    return False


def _validate_ip_str(ip_str: str, source: str) -> Tuple[bool, Optional[str]]:
    """Return (False, error) if *ip_str* resolves to a forbidden address."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return True, None
    if _is_forbidden_address(addr):
        return (
            False,
            f"URLs pointing to private/internal networks are not allowed: {source}",
        )
    return True, None


def _resolve_and_validate_dns(
    hostname: str,
    strict_dns: Optional[bool] = None,
) -> Tuple[bool, Optional[str]]:
    """Resolve *hostname* via DNS and reject any private/internal addresses."""
    if strict_dns is None:
        strict_dns = config.MISTRAL_DOCUMENT_URL_STRICT_DNS
    try:
        _future = _dns_executor.submit(socket.getaddrinfo, hostname, None, proto=socket.IPPROTO_TCP)
        try:
            infos = _future.result(timeout=config.MISTRAL_DOCUMENT_URL_DNS_TIMEOUT_SECONDS)
        except DnsTimeoutError:
            # Best-effort cancel so the worker thread returns to the pool instead
            # of staying blocked in getaddrinfo(). The C call is not actually
            # interruptible, but marking the future cancelled keeps the pool
            # clean for future callers.
            _future.cancel()
            raise socket.timeout("DNS resolution timed out")
        resolved_ips = {str(info[4][0]) for info in infos if info and info[4]}
        for ip in resolved_ips:
            ok, err = _validate_ip_str(ip, f"{hostname} -> {ip}")
            if not ok:
                return ok, err
    except socket.gaierror:
        if strict_dns:
            return (
                False,
                f"Could not resolve hostname in strict document URL mode: {hostname}",
            )
        logger.debug("Could not resolve hostname during SSRF validation: %s", hostname)
    except socket.timeout:
        if strict_dns:
            return (
                False,
                f"DNS resolution timed out in strict document URL mode: {hostname}",
            )
        logger.debug("DNS resolution timed out for %s", hostname)
    except (OSError, socket.error) as e:
        if strict_dns:
            return (
                False,
                f"DNS resolution check failed in strict document URL mode for {hostname}: {e}",
            )
        logger.debug("DNS resolution check skipped for %s: %s", hostname, e)

    return True, None


def _validate_document_url(
    url: str,
    strict_dns: Optional[bool] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate a document URL to prevent SSRF attacks.

    Rejects non-HTTPS URLs, URLs with embedded credentials, and URLs
    pointing to private/internal networks (IPv4 and IPv6).

    Args:
        url: URL to validate
        strict_dns: When True, DNS lookup failures reject the URL. When None,
            uses ``config.MISTRAL_DOCUMENT_URL_STRICT_DNS``.

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        parsed = attr("urlparse")(url)
    except ValueError:
        return False, "Invalid URL format"

    if parsed.scheme not in ("https",):
        return False, f"Only HTTPS URLs are allowed (got {parsed.scheme}://)"

    if parsed.username or parsed.password:
        return False, "URLs with embedded credentials are not allowed"

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return False, "URL must include a hostname"

    # These are SSRF blocklist entries, not bind addresses. Bandit's B104 flags
    # "0.0.0.0" on sight because it is commonly mis-used to bind a server to all
    # interfaces; here the string is just one of several internal hostnames /
    # IPs we want to REJECT in document URLs. The nosec scope is intentionally
    # narrow.
    blocked_hosts = {
        "localhost",
        "127.0.0.1",
        "0.0.0.0",  # nosec B104 - blocklist entry, not a bind address
        "::1",
        "[::1]",
        "metadata.google.internal",
        "169.254.169.254",
        "metadata.google.internal.",
    }
    if hostname in blocked_hosts:
        return False, f"URLs pointing to internal hosts are not allowed: {hostname}"

    ok, err = _validate_ip_str(hostname.strip("[]"), hostname)
    if not ok:
        return ok, err

    return _resolve_and_validate_dns(hostname, strict_dns=strict_dns)


def validate_https_document_url(
    url: str,
    strict_dns: Optional[bool] = None,
) -> Tuple[bool, Optional[str]]:
    """Public SSRF-safe HTTPS URL check for Document QnA."""
    return _validate_document_url(url, strict_dns=strict_dns)
