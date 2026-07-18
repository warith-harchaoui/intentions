"""Capture the GUI screenshots used in the docs, bilingual + both themes.

Module summary
--------------
Drives the running FastAPI app with a headless Chromium to produce the doc
figures under ``docs/img/``. Language (fr/en) and theme (light/dark) are
seeded via ``localStorage`` before the page boots, so the app renders in the
requested locale/theme with no flicker.

Shots
-----
* ``accueil`` / ``home`` — landing hero + the five approach cards + input.
* ``comparateur`` — a query classified across all engines (needs the LLM, so
  Ollama must be up); run with ``--comparator`` once the shootout is idle.
* ``mode-sombre`` — the landing in dark mode.
* ``base-connaissance`` — the knowledge-base browser, expanded.

Usage
-----
    python scripts/screenshots.py --base-url http://localhost:8077
    python scripts/screenshots.py --comparator   # the LLM-dependent shot

Author
------
Project maintainers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

_IMG_DIR = Path(__file__).resolve().parent.parent / "docs" / "img"

# A vivid, realistic French query that every engine can classify (a car
# accident with an extractable urgency slot for the LLM to show off).
_QUERY_FR = "j'ai eu un accident ce matin, ma voiture est bien cabossée"
_QUERY_EN = "I had an accident this morning, my car is badly dented"


def _new_page(pw, lang: str, dark: bool, width: int = 1360, height: int = 900):
    """Open a page pre-seeded with the requested language and theme.

    Parameters
    ----------
    pw : Playwright
        Active Playwright instance.
    lang : str
        ``"fr"`` or ``"en"`` — written to ``localStorage`` before boot.
    dark : bool
        Whether to seed dark mode.
    width, height : int
        Viewport size (2× device scale for crisp PNGs).

    Returns
    -------
    tuple
        ``(browser, page)`` — caller closes the browser.
    """
    browser = pw.chromium.launch()
    page = browser.new_page(
        viewport={"width": width, "height": height}, device_scale_factor=2.0
    )
    # Seed the locale + theme the same way a returning user would have them,
    # BEFORE any page script runs, so there is no flash of the wrong language.
    theme = "dark" if dark else "light"
    page.add_init_script(
        f"localStorage.setItem('lang', '{lang}');"
        f"localStorage.setItem('theme', '{theme}');"
    )
    return browser, page


def _settle(page, base_url: str, lang: str) -> None:
    """Navigate and wait until the app has applied i18n and loaded the KB."""
    page.goto(base_url, wait_until="networkidle")
    # The KB list is populated by JS after /api/kb; its presence means the app
    # booted and applied translations. It lives inside a collapsed <details>,
    # so wait for it to be ATTACHED (not visible).
    page.wait_for_selector("#kb-list > div", state="attached", timeout=20_000)
    # A beat for fonts + the i18n pass to paint.
    page.wait_for_timeout(500)
    # Sanity: <html lang> must match the requested locale.
    page.wait_for_function(f"document.documentElement.lang === '{lang}'", timeout=5_000)


def shoot_static(base_url: str) -> list[Path]:
    """Capture the shots that need no classification (no Ollama call).

    Parameters
    ----------
    base_url : str
        The running app's base URL.

    Returns
    -------
    list[Path]
        Written PNG paths.
    """
    from playwright.sync_api import sync_playwright

    _IMG_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with sync_playwright() as pw:
        # Landing (light) in both languages.
        for lang, name in (("fr", "01-accueil"), ("en", "01-accueil-en")):
            browser, page = _new_page(pw, lang, dark=False)
            _settle(page, base_url, lang)
            out = _IMG_DIR / f"{name}.png"
            page.screenshot(path=str(out))
            written.append(out)
            browser.close()

        # Dark mode, both languages (the theme is language-agnostic, but each
        # doc variant shows its own chrome).
        for lang, name in (("fr", "03-mode-sombre"), ("en", "03-mode-sombre-en")):
            browser, page = _new_page(pw, lang, dark=True)
            _settle(page, base_url, lang)
            out = _IMG_DIR / f"{name}.png"
            page.screenshot(path=str(out))
            written.append(out)
            browser.close()

        # Knowledge-base browser, expanded, both languages.
        for lang, name in (
            ("fr", "04-base-connaissance"),
            ("en", "04-base-connaissance-en"),
        ):
            browser, page = _new_page(pw, lang, dark=False, height=1100)
            _settle(page, base_url, lang)
            # Expand the <details> and bring it into view.
            page.evaluate("document.querySelector('details').open = true")
            page.wait_for_timeout(300)
            details = page.query_selector("details")
            details.scroll_into_view_if_needed()
            out = _IMG_DIR / f"{name}.png"
            details.screenshot(path=str(out))
            written.append(out)
            browser.close()
    return written


def shoot_comparator(base_url: str) -> list[Path]:
    """Capture the 5-engine comparator (submits a query — needs Ollama up).

    Parameters
    ----------
    base_url : str
        The running app's base URL.

    Returns
    -------
    list[Path]
        Written PNG paths.
    """
    from playwright.sync_api import sync_playwright

    _IMG_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with sync_playwright() as pw:
        for lang, query, name in (
            ("fr", _QUERY_FR, "02-comparateur-5-moteurs"),
            ("en", _QUERY_EN, "02-comparateur-5-moteurs-en"),
        ):
            browser, page = _new_page(pw, lang, dark=False, height=1000)
            _settle(page, base_url, lang)
            # Type the query and submit.
            page.fill("#query", query)
            page.click("#submit-btn")
            # Wait for the REAL engine cards — the loading skeletons are also
            # <article> elements (class ``animate-pulse``), so exclude them; and
            # wait for the execution panel to un-hide, which happens only after
            # every engine (the slow LLM included) has answered.
            page.wait_for_function(
                "document.querySelectorAll('#results article:not(.animate-pulse)')"
                ".length >= 5",
                timeout=120_000,
            )
            page.wait_for_selector("#execution:not(.hidden)", timeout=120_000)
            page.wait_for_timeout(500)
            page.evaluate(
                "window.scrollTo(0, document.querySelector('#results').offsetTop - 80)"
            )
            out = _IMG_DIR / f"{name}.png"
            page.screenshot(path=str(out))
            written.append(out)
            browser.close()
    return written


def main(argv: list[str] | None = None) -> int:
    """CLI: capture the doc screenshots.

    Parameters
    ----------
    argv : list[str] | None, optional
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(prog="screenshots")
    parser.add_argument("--base-url", default="http://localhost:8077")
    parser.add_argument(
        "--comparator",
        action="store_true",
        help="Only the 5-engine comparator shot (needs Ollama up).",
    )
    parser.add_argument(
        "--all", action="store_true", help="Static shots AND the comparator."
    )
    args = parser.parse_args(argv)

    written: list[Path] = []
    if args.comparator:
        written += shoot_comparator(args.base_url)
    elif args.all:
        written += shoot_static(args.base_url)
        written += shoot_comparator(args.base_url)
    else:
        written += shoot_static(args.base_url)

    for path in written:
        print(f"Capture : {path}")
    return 0 if written else 1


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
