"""Tests for SSRF URL validation helpers in mistral_converter.

Covers _validate_document_url, _is_forbidden_address, DNS resolution,
and the signed-URL expiry classifier used by the QnA retry path.
Split out of test_mistral_converter.py for navigability."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

import config

# Initialize config dirs so imports work
config.ensure_directories()

import mistral_converter
import mistral_converter.url_validation as url_module

# ============================================================================
# _validate_document_url Tests
# ============================================================================


class TestValidateDocumentUrl:
    """Test SSRF prevention in _validate_document_url."""

    def test_valid_https_url(self):
        with patch.object(url_module, "_resolve_dns_in_subprocess", return_value=["93.184.216.34"]):
            ok, err = url_module._validate_document_url("https://example.com/doc.pdf")
        assert ok is True
        assert err is None

    def test_rejects_hostname_resolving_to_loopback(self):
        with patch.object(url_module, "_resolve_dns_in_subprocess", return_value=["127.0.0.1"]):
            ok, err = url_module._validate_document_url("https://localtest.me/doc.pdf")
        assert ok is False
        assert "internal" in err.lower()

    def test_rejects_http(self):
        ok, err = url_module._validate_document_url("http://example.com/doc.pdf")
        assert ok is False
        assert "HTTPS" in err

    def test_rejects_ftp(self):
        ok, err = url_module._validate_document_url("ftp://example.com/doc.pdf")
        assert ok is False

    def test_rejects_localhost(self):
        ok, err = url_module._validate_document_url("https://localhost/secret")
        assert ok is False
        assert "internal" in err.lower()

    def test_rejects_127_0_0_1(self):
        ok, err = url_module._validate_document_url("https://127.0.0.1/admin")
        assert ok is False

    def test_rejects_ipv4_private_10(self):
        ok, err = url_module._validate_document_url("https://10.0.0.1/")
        assert ok is False
        assert "private" in err.lower()

    def test_rejects_ipv4_private_172(self):
        ok, err = url_module._validate_document_url("https://172.16.0.1/")
        assert ok is False

    def test_rejects_ipv4_private_192(self):
        ok, err = url_module._validate_document_url("https://192.168.1.1/")
        assert ok is False

    def test_rejects_ipv6_loopback(self):
        ok, err = url_module._validate_document_url("https://[::1]/")
        assert ok is False

    def test_rejects_cloud_metadata(self):
        ok, err = url_module._validate_document_url("https://169.254.169.254/latest/meta-data/")
        assert ok is False

    def test_rejects_embedded_credentials(self):
        ok, err = url_module._validate_document_url("https://user:pass@example.com/doc.pdf")
        assert ok is False
        assert "credentials" in err.lower()

    def test_rejects_empty_hostname(self):
        ok, err = url_module._validate_document_url("https:///path")
        assert ok is False

    def test_accepts_public_ip(self):
        ok, err = url_module._validate_document_url("https://8.8.8.8/doc.pdf")
        assert ok is True

    def test_rejects_ipv6_private(self):
        ok, err = url_module._validate_document_url("https://[fd12::1]/doc.pdf")
        assert ok is False

    def test_rejects_ipv4_mapped_ipv6_loopback(self):
        ok, err = url_module._validate_document_url("https://[::ffff:127.0.0.1]/")
        assert ok is False

    def test_rejects_link_local(self):
        ok, err = url_module._validate_document_url("https://169.254.1.1/")
        assert ok is False

    def test_lenient_dns_allows_when_resolve_fails(self, monkeypatch):
        import socket

        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_URL_STRICT_DNS", False)
        with patch.object(
            url_module, "_resolve_dns_in_subprocess", side_effect=socket.gaierror(8, "nodename nor servname")
        ):
            ok, err = url_module._validate_document_url("https://example.com/doc.pdf")
        assert ok is True
        assert err is None

    def test_strict_dns_rejects_when_resolve_fails(self, monkeypatch):
        import socket

        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_URL_STRICT_DNS", True)
        with patch.object(
            url_module, "_resolve_dns_in_subprocess", side_effect=socket.gaierror(8, "nodename nor servname")
        ):
            ok, err = url_module._validate_document_url("https://example.com/doc.pdf")
        assert ok is False
        assert err and "strict" in err.lower()


class TestResolverSubprocess:
    """The DNS timeout boundary must be process-owned, not a lingering thread."""

    def test_resolver_passes_hostname_without_a_shell(self):
        completed = MagicMock(returncode=0, stdout='["93.184.216.34"]', stderr="")
        with patch.object(url_module.subprocess, "run", return_value=completed) as run:
            assert url_module._resolve_dns_in_subprocess("example.com") == ["93.184.216.34"]
        args, kwargs = run.call_args
        assert args[0][-1] == "example.com"
        assert kwargs["timeout"] == config.MISTRAL_DOCUMENT_URL_DNS_TIMEOUT_SECONDS
        assert "shell" not in kwargs

    def test_resolver_timeout_is_reported_to_strict_validation(self, monkeypatch):
        import socket

        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_URL_STRICT_DNS", True)
        with patch.object(url_module, "_resolve_dns_in_subprocess", side_effect=socket.timeout()):
            valid, error = url_module._validate_document_url("https://example.com/doc.pdf")
        assert valid is False
        assert "timed out" in (error or "").lower()

    def test_resolver_rejects_when_process_capacity_is_exhausted(self, monkeypatch):
        import socket

        slots = MagicMock()
        slots.acquire.return_value = False
        monkeypatch.setattr(url_module, "_dns_process_slots", slots)
        with patch.object(url_module.subprocess, "run") as run:
            with pytest.raises(socket.timeout, match="capacity"):
                url_module._resolve_dns_in_subprocess("example.com")
        run.assert_not_called()
        slots.release.assert_not_called()


class TestValidateHttpsDocumentUrlPublic:
    """Public SSRF API must stay aligned with _validate_document_url."""

    def test_delegates_to_private_validator(self):
        # Hosts blocked without DNS so results are deterministic.
        for url in ("https://localhost/secret", "https://10.0.0.1/"):
            a, ea = url_module._validate_document_url(url)
            b, eb = mistral_converter.validate_https_document_url(url)
            assert (a, ea) == (b, eb)


# ============================================================================
# _is_weak_page Tests
# ============================================================================


class TestValidateDocumentUrlAdditional:
    """Additional URL validation edge cases."""

    def test_rejects_data_url(self):
        ok, err = url_module._validate_document_url("data:text/html,<h1>bad</h1>")
        assert ok is False

    def test_rejects_javascript_url(self):
        ok, err = url_module._validate_document_url("javascript:alert(1)")
        assert ok is False

    def test_rejects_empty_url(self):
        ok, err = url_module._validate_document_url("")
        assert ok is False

    def test_rejects_non_string(self):
        ok, err = url_module._validate_document_url(None)
        assert ok is False


# ============================================================================
# get_mistral_client Tests
# ============================================================================


class TestValidateDocumentUrlSSRFEdges:
    """Lines 1681-1683, 1697-1698, 1739-1742: SSRF edge cases."""

    def test_ipv6_mapped_private_ip(self):
        """Lines 1681-1683: IPv6-mapped IPv4 private address."""
        import socket

        with patch.object(url_module, "_resolve_dns_in_subprocess", return_value=["::ffff:127.0.0.1"]):
            valid, err = url_module._validate_document_url("https://example.com/doc.pdf")
        assert valid is False
        assert "private" in err.lower() or "internal" in err.lower()

    def test_dns_resolution_other_exception(self, monkeypatch):
        """Non-gaierror DNS exception: lenient mode defers to upstream."""
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_URL_STRICT_DNS", False)
        with patch.object(url_module, "_resolve_dns_in_subprocess", side_effect=OSError("DNS service unavailable")):
            valid, err = url_module._validate_document_url("https://example.com/doc.pdf")
        assert valid is True

    def test_dns_gaierror(self, monkeypatch):
        """socket.gaierror during resolution: lenient mode defers to upstream."""
        import socket

        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_URL_STRICT_DNS", False)
        with patch.object(
            url_module, "_resolve_dns_in_subprocess", side_effect=socket.gaierror("Name resolution failed")
        ):
            valid, err = url_module._validate_document_url("https://example.com/doc.pdf")
        assert valid is True


# ============================================================================
# query_document - DNS resolution edge cases
# ============================================================================


class TestQueryDocumentDNS:
    """Lines 1739-1742: query_document DNS resolution paths."""

    def test_dns_gaierror_still_proceeds(self, monkeypatch):
        """DNS resolution fails -> defers to Mistral when strict DNS is off."""
        import socket

        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_URL_STRICT_DNS", False)
        monkeypatch.setattr(config, "MISTRAL_DOCUMENT_QNA_MODEL", "mistral-small-latest")
        monkeypatch.setattr(config, "MISTRAL_QNA_SYSTEM_PROMPT", "")
        monkeypatch.setattr(config, "MISTRAL_QNA_DOCUMENT_IMAGE_LIMIT", 0)
        monkeypatch.setattr(config, "MISTRAL_QNA_DOCUMENT_PAGE_LIMIT", 0)

        mock_choice = MagicMock()
        mock_choice.message.content = "Answer"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.complete.return_value = mock_response

        with patch.object(mistral_converter, "get_mistral_client", return_value=mock_client):
            with patch.object(mistral_converter, "get_retry_config", return_value=None):
                with patch.object(url_module, "_resolve_dns_in_subprocess", side_effect=socket.gaierror("fail")):
                    ok, answer, err = mistral_converter.query_document("https://example.com/doc.pdf", "What?")

        assert ok is True
        assert answer == "Answer"


# ============================================================================
# _extract_response_metadata - dict response paths
# ============================================================================


class TestValidateUrlIpv6MappedMocked:
    """Lines 1681-1683: IPv6-mapped IPv4 private addr via mocked ip_address."""

    def test_ipv6_mapped_private_mocked(self):
        import ipaddress as real_ipaddress
        import socket

        real_ip_addr = real_ipaddress.ip_address

        def custom_ip_address(ip_str):
            if ip_str == "::ffff:10.0.0.1":
                mock = MagicMock(spec=real_ipaddress.IPv6Address)
                mock.is_private = False
                mock.is_reserved = False
                mock.is_loopback = False
                mock.is_link_local = False
                mock.is_multicast = False
                mapped = real_ip_addr("10.0.0.1")
                mock.ipv4_mapped = mapped
                return mock
            return real_ip_addr(ip_str)

        with patch("ipaddress.ip_address", side_effect=custom_ip_address):
            with patch.object(url_module, "_resolve_dns_in_subprocess", return_value=["::ffff:10.0.0.1"]):
                valid, err = url_module._validate_document_url("https://example.com/doc.pdf")

        assert valid is False
        assert "private" in err.lower() or "internal" in err.lower()


class TestValidateUrlParseException:
    """Lines 1697-1698: urlparse exception."""

    def test_urlparse_raises(self):
        with patch(
            "mistral_converter.url_validation.attr",
            side_effect=ValueError("parse error"),
        ):
            valid, err = url_module._validate_document_url("https://example.com/doc.pdf")

        assert valid is False
        assert "Invalid URL format" in err


class TestIsForbiddenAddress:
    """_is_forbidden_address with various IP types."""

    def test_loopback_is_forbidden(self):
        import ipaddress

        assert url_module._is_forbidden_address(ipaddress.ip_address("127.0.0.1")) is True

    def test_private_is_forbidden(self):
        import ipaddress

        assert url_module._is_forbidden_address(ipaddress.ip_address("192.168.1.1")) is True

    def test_multicast_is_forbidden(self):
        import ipaddress

        assert url_module._is_forbidden_address(ipaddress.ip_address("224.0.0.1")) is True

    def test_public_is_allowed(self):
        import ipaddress

        assert url_module._is_forbidden_address(ipaddress.ip_address("8.8.8.8")) is False

    def test_ipv6_loopback_forbidden(self):
        import ipaddress

        assert url_module._is_forbidden_address(ipaddress.ip_address("::1")) is True

    def test_ipv4_mapped_ipv6_loopback_forbidden(self):
        import ipaddress

        addr = ipaddress.ip_address("::ffff:127.0.0.1")
        assert url_module._is_forbidden_address(addr) is True


class TestIsSignedUrlExpiryError:
    """Cover the replacement for the old substring-based retry heuristic."""

    def test_empty_message_is_not_expiry(self):
        assert mistral_converter.is_signed_url_expiry_error(None) is False
        assert mistral_converter.is_signed_url_expiry_error("") is False

    def test_random_url_mention_is_not_expiry(self):
        # The previous heuristic incorrectly retried on any message containing
        # "url"; the new classifier must not trigger on these.
        assert mistral_converter.is_signed_url_expiry_error("Failed to resolve document URL for QnA") is False
        assert mistral_converter.is_signed_url_expiry_error("QnA stream failed: network reset") is False

    def test_permanent_auth_not_retried(self):
        assert (
            mistral_converter.is_signed_url_expiry_error("Mistral API authentication failed (401 Unauthorized).")
            is False
        )
        assert mistral_converter.is_signed_url_expiry_error("Invalid API key for workspace") is False

    def test_signed_url_expiry_detected(self):
        assert mistral_converter.is_signed_url_expiry_error("The signed URL has expired") is True
        assert mistral_converter.is_signed_url_expiry_error("403 Forbidden") is False
        assert mistral_converter.is_signed_url_expiry_error("403 Forbidden: signed URL has expired") is True
        assert mistral_converter.is_signed_url_expiry_error("Failed to fetch document from URL") is True
        assert mistral_converter.is_signed_url_expiry_error("Signature mismatch") is True
        assert mistral_converter.is_signed_url_expiry_error("Access denied") is False

    def test_auth_hint_overrides_expiry_hint(self):
        # Permanent auth hints must win over expiry hints if both appear.
        msg = "401 Unauthorized - signed URL"
        assert mistral_converter.is_signed_url_expiry_error(msg) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
