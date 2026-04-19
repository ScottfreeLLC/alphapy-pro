"""
Substack publishing client using direct API calls.
Handles draft creation, prepublish, and full publish workflow.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from config import (
    SUBSTACK_COOKIES_PATH,
    SUBSTACK_EMAIL,
    SUBSTACK_PASSWORD,
    SUBSTACK_PUBLICATION_URL,
)

logger = logging.getLogger(__name__)


class SubstackPublisher:
    """
    Publishes posts to Substack via its internal API.

    Auth methods (in order of preference):
    1. Cookies file (exported from browser)
    2. Email/password login

    Three-step workflow: create_draft() -> prepublish() -> publish()
    Or use full_publish() for convenience.
    """

    def __init__(
        self,
        publication_url: str = "",
        email: str = "",
        password: str = "",
        cookies_path: str = "",
    ):
        self.publication_url = (publication_url or SUBSTACK_PUBLICATION_URL).rstrip("/")
        self.email = email or SUBSTACK_EMAIL
        self.password = password or SUBSTACK_PASSWORD
        self.cookies_path = cookies_path or SUBSTACK_COOKIES_PATH
        self._client: Optional[httpx.AsyncClient] = None
        self._authenticated = False
        self._subdomain = ""

        if self.publication_url:
            # Extract subdomain: "https://foo.substack.com" -> "foo"
            import re
            match = re.search(r"https?://([^.]+)\.substack\.com", self.publication_url)
            if match:
                self._subdomain = match.group(1)

    def is_configured(self) -> bool:
        """Check if Substack credentials are configured."""
        has_auth = bool(self.cookies_path) or (bool(self.email) and bool(self.password))
        return bool(self.publication_url) and has_auth

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create an authenticated HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Alfi/1.0",
                    "Content-Type": "application/json",
                },
                follow_redirects=True,
            )

        if not self._authenticated:
            await self._authenticate()

        return self._client

    async def _authenticate(self):
        """Authenticate via cookies file or email/password."""
        if self.cookies_path:
            await self._auth_cookies()
        elif self.email and self.password:
            await self._auth_login()
        else:
            raise ValueError("No Substack auth method configured")

    async def _auth_cookies(self):
        """Load auth from a cookies file (Netscape format or JSON)."""
        try:
            with open(self.cookies_path, "r") as f:
                content = f.read().strip()

            cookies = {}
            if content.startswith("[") or content.startswith("{"):
                # JSON format
                data = json.loads(content)
                if isinstance(data, list):
                    for c in data:
                        cookies[c["name"]] = c["value"]
                elif isinstance(data, dict):
                    cookies = data
            else:
                # Netscape format
                for line in content.split("\n"):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        parts = line.split("\t")
                        if len(parts) >= 7:
                            cookies[parts[5]] = parts[6]

            if cookies:
                self._client.cookies.update(cookies)
                self._authenticated = True
                logger.info("Substack: authenticated via cookies")
            else:
                raise ValueError("No cookies found in file")

        except Exception as e:
            logger.error(f"Cookie auth failed: {e}")
            raise

    async def _auth_login(self):
        """Authenticate via email/password."""
        try:
            resp = await self._client.post(
                "https://substack.com/api/v1/login",
                json={
                    "email": self.email,
                    "password": self.password,
                    "redirect": self.publication_url,
                },
            )
            resp.raise_for_status()
            self._authenticated = True
            logger.info("Substack: authenticated via email/password")
        except Exception as e:
            logger.error(f"Login auth failed: {e}")
            raise

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ---- Publishing workflow ----

    async def create_draft(
        self,
        title: str,
        body_html: str,
        subtitle: str = "",
        audience: str = "everyone",
    ) -> Dict[str, Any]:
        """
        Create a draft post on Substack.

        Args:
            title: Post title
            body_html: HTML body content
            subtitle: Optional subtitle
            audience: "everyone" (free) or "only_paid"

        Returns:
            Draft metadata dict with 'id', 'slug', etc.
        """
        client = await self._get_client()
        api_base = f"{self.publication_url}/api/v1"

        body = {
            "draft_title": title,
            "draft_subtitle": subtitle,
            "draft_body": self._wrap_body(body_html),
            "audience": audience,
            "type": "newsletter",
        }

        try:
            resp = await client.post(f"{api_base}/drafts", json=body)
            resp.raise_for_status()
            draft = resp.json()
            logger.info(f"Created draft: {draft.get('id')} - {title}")
            return draft
        except Exception as e:
            logger.error(f"Failed to create draft: {e}")
            raise

    async def prepublish(self, draft_id: int) -> Dict[str, Any]:
        """Prepare a draft for publishing (validates, generates preview)."""
        client = await self._get_client()
        api_base = f"{self.publication_url}/api/v1"

        try:
            resp = await client.post(f"{api_base}/drafts/{draft_id}/prepublish")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Prepublish failed for draft {draft_id}: {e}")
            raise

    async def publish(self, draft_id: int, send_email: bool = True) -> Dict[str, Any]:
        """Publish a draft."""
        client = await self._get_client()
        api_base = f"{self.publication_url}/api/v1"

        try:
            resp = await client.post(
                f"{api_base}/drafts/{draft_id}/publish",
                json={"send": send_email},
            )
            resp.raise_for_status()
            result = resp.json()
            logger.info(f"Published draft {draft_id}")
            return result
        except Exception as e:
            logger.error(f"Publish failed for draft {draft_id}: {e}")
            raise

    async def full_publish(
        self,
        title: str,
        body_html: str,
        subtitle: str = "",
        audience: str = "everyone",
        send_email: bool = True,
    ) -> Dict[str, Any]:
        """Convenience: create draft -> prepublish -> publish in one call."""
        draft = await self.create_draft(title, body_html, subtitle, audience)
        draft_id = draft["id"]
        await self.prepublish(draft_id)
        return await self.publish(draft_id, send_email)

    async def get_drafts(self) -> List[Dict]:
        """List existing drafts."""
        client = await self._get_client()
        api_base = f"{self.publication_url}/api/v1"

        try:
            resp = await client.get(f"{api_base}/drafts")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to list drafts: {e}")
            return []

    @staticmethod
    def _wrap_body(html: str) -> str:
        """Wrap raw HTML in Substack's expected body JSON format."""
        # Substack expects a ProseMirror-like JSON doc or raw HTML.
        # For simplicity, we send HTML which Substack handles.
        return html

    @staticmethod
    def markdown_to_html(markdown_text: str) -> str:
        """Convert markdown to HTML for Substack."""
        # Basic markdown-to-HTML conversion without external deps
        import re
        html = markdown_text

        # Headers
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

        # Bold and italic
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

        # Links
        html = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', html)

        # Line breaks to paragraphs
        paragraphs = html.split('\n\n')
        html = ''.join(
            f'<p>{p.strip()}</p>' if not p.strip().startswith('<h') else p.strip()
            for p in paragraphs if p.strip()
        )

        return html
