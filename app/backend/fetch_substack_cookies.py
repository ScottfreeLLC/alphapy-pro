"""
Fetch Substack session cookies via automated Playwright login.

Usage:
    uv run python app/backend/fetch_substack_cookies.py

Reads SUBSTACK_EMAIL and SUBSTACK_PASSWORD from .env, logs in via
headless Chromium, and saves cookies to substack_cookies.json.
Updates .env with SUBSTACK_COOKIES_PATH automatically.
"""

import json
import os
import re
import sys

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

load_dotenv()

COOKIES_FILENAME = "substack_cookies.json"


def fetch_cookies():
    email = os.getenv("SUBSTACK_EMAIL", "")
    password = os.getenv("SUBSTACK_PASSWORD", "")

    if not email or not password:
        print("ERROR: Set SUBSTACK_EMAIL and SUBSTACK_PASSWORD in .env first.")
        sys.exit(1)

    print(f"Logging into Substack as {email}...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        # Navigate to sign-in
        page.goto("https://substack.com/sign-in", wait_until="networkidle")

        # Click "Sign in with password" to reveal the email/password form
        try:
            signin_link = page.get_by_text("Sign in with password")
            signin_link.click()
            page.wait_for_timeout(1000)
        except Exception:
            pass  # Form may already be visible

        # Fill credentials
        page.fill('input[name="email"], input[type="email"]', email)
        page.fill('input[name="password"], input[type="password"]', password)

        # Submit
        page.click('button[type="submit"], button:has-text("Sign in")')

        # Wait for navigation after login (redirect to dashboard or homepage)
        try:
            page.wait_for_url("**/inbox**", timeout=15000)
        except Exception:
            # May redirect elsewhere — wait a bit and check cookies anyway
            page.wait_for_timeout(5000)

        # Check if we're still on the sign-in page (login failed)
        if "sign-in" in page.url:
            print("ERROR: Login failed. Check your email/password.")
            print("  If Substack requires CAPTCHA, use the interactive method instead.")
            browser.close()
            sys.exit(1)

        # Extract cookies
        cookies = context.cookies()
        browser.close()

    # Filter to Substack cookies
    substack_cookies = [
        c for c in cookies
        if "substack.com" in c.get("domain", "")
    ]

    if not substack_cookies:
        print("ERROR: No Substack cookies found after login.")
        sys.exit(1)

    # Save cookies
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cookies_path = os.path.join(script_dir, COOKIES_FILENAME)

    with open(cookies_path, "w") as f:
        json.dump(substack_cookies, f, indent=2)

    print(f"Saved {len(substack_cookies)} cookies to {cookies_path}")

    # Update .env with cookies path
    env_path = os.path.join(script_dir, ".env")
    _update_env(env_path, "SUBSTACK_COOKIES_PATH", cookies_path)

    print("Done! SUBSTACK_COOKIES_PATH updated in .env")


def _update_env(env_path: str, key: str, value: str):
    """Add or update a key in the .env file."""
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write(f"{key}={value}\n")
        return

    with open(env_path, "r") as f:
        content = f.read()

    # Replace existing (commented or uncommented)
    pattern = rf"^#?\s*{re.escape(key)}=.*$"
    replacement = f"{key}={value}"

    if re.search(pattern, content, re.MULTILINE):
        content = re.sub(pattern, replacement, content, count=1, flags=re.MULTILINE)
    else:
        content = content.rstrip("\n") + f"\n{key}={value}\n"

    with open(env_path, "w") as f:
        f.write(content)


if __name__ == "__main__":
    fetch_cookies()
