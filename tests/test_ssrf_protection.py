import asyncio
import pytest
from unittest.mock import patch, MagicMock

from mcp_server.research_tools import (
    validate_url_for_ssrf,
    fetch_webpage_markdown,
    is_safe_ip_address,
)
import ipaddress


@pytest.mark.asyncio
async def test_scheme_validation():
    """Non-http(s) schemes must be rejected immediately."""
    for bad_url in [
        "ftp://example.com/file",
        "file:///etc/passwd",
        "gopher://evil.com",
        "javascript:alert(1)",
    ]:
        safe, err, _ = await validate_url_for_ssrf(bad_url)
        assert not safe
        assert "Only http:// and https:// are permitted" in err


@pytest.mark.asyncio
async def test_ip_range_restrictions():
    """Validates rejection of loopback, private, link-local, multicast, and IPv6 unique-local."""
    bad_ips = [
        "127.0.0.1",
        "127.0.1.1",
        "::1",
        "10.0.0.1",
        "10.254.0.1",
        "172.16.0.1",
        "172.31.255.255",
        "192.168.1.1",
        "169.254.169.254",  # AWS/GCP/Azure instance metadata
        "169.254.1.1",
        "224.0.0.1",        # Multicast
        "0.0.0.0",          # Unspecified
        "::",
        "fc00::1",          # IPv6 Unique Local Address
        "fd12:3456:789a::1",
    ]
    for ip_str in bad_ips:
        ip_obj = ipaddress.ip_address(ip_str)
        safe, reason = is_safe_ip_address(ip_obj)
        assert not safe, f"Expected {ip_str} to be rejected, but marked safe ({reason})"

    # Legitimate public IP must be allowed
    public_ip = ipaddress.ip_address("93.184.216.34")  # example.com
    safe, reason = is_safe_ip_address(public_ip)
    assert safe, f"Expected {public_ip} to be safe, rejected due to {reason}"


@pytest.mark.asyncio
async def test_ssrf_rejects_loopback_and_metadata_hostnames():
    """Ensures hostnames resolving to private/metadata IPs are blocked."""
    # Localhost resolves to 127.0.0.1
    safe, err, _ = await validate_url_for_ssrf("http://localhost:8080/admin")
    assert not safe
    assert "rejected" in err

    safe_ip, err_ip, _ = await validate_url_for_ssrf("http://169.254.169.254/computeMetadata/v1/")
    assert not safe_ip
    assert "rejected" in err_ip


@pytest.mark.asyncio
async def test_redirect_to_internal_metadata_blocked():
    """An external redirect leading to a private/metadata IP must be blocked on the redirect hop."""
    from unittest.mock import AsyncMock
    with patch("mcp_server.research_tools.validate_url_for_ssrf") as mock_validate:
        mock_validate.side_effect = [
            (True, "", None),
            (False, "Access to '169.254.169.254' rejected: link-local address (SSRF protection).", None),
        ]

        with patch("httpx.AsyncClient.send", new_callable=AsyncMock) as mock_send:
            mock_resp = MagicMock()
            mock_resp.status_code = 302
            mock_resp.headers = {"location": "http://169.254.169.254/latest/meta-data/"}
            mock_resp.aclose = AsyncMock()
            mock_send.return_value = mock_resp

            result = await fetch_webpage_markdown("http://example.com/redirect")
            assert result["status"] == "error"
            assert "SSRF protection" in result["error"]


@pytest.mark.asyncio
async def test_max_redirect_hops_exceeded():
    """Redirect loops or chains exceeding 3 hops must abort with error."""
    from unittest.mock import AsyncMock
    with patch("mcp_server.research_tools.validate_url_for_ssrf", return_value=(True, "", None)):
        with patch("httpx.AsyncClient.send", new_callable=AsyncMock) as mock_send:
            mock_resp = MagicMock()
            mock_resp.status_code = 301
            mock_resp.headers = {"location": "http://example.com/next-hop"}
            mock_resp.aclose = AsyncMock()
            mock_send.return_value = mock_resp

            result = await fetch_webpage_markdown("http://example.com/loop")
            assert result["status"] == "error"
            assert "Too many redirects" in result["error"]


@pytest.mark.asyncio
async def test_response_size_limit_exceeded(monkeypatch):
    """Responses larger than MAX_WEB_PAGE_BYTES must abort."""
    from unittest.mock import AsyncMock
    monkeypatch.setenv("MAX_WEB_PAGE_BYTES", "1024")  # 1 KB limit

    with patch("mcp_server.research_tools.validate_url_for_ssrf", return_value=(True, "", None)):
        with patch("httpx.AsyncClient.send", new_callable=AsyncMock) as mock_send:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {"content-length": "2048"}  # Claims 2 KB
            mock_resp.aclose = AsyncMock()
            mock_send.return_value = mock_resp

            result = await fetch_webpage_markdown("http://example.com/big-file")
            assert result["status"] == "error"
            assert "Response size exceeds maximum limit" in result["error"]
