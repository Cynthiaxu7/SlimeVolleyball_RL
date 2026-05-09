#!/usr/bin/env python3
"""Capture publication-quality screenshots of the Slime Volleyball web demo.

Produces four PNGs in ``report/figures/``:

* ``web_freeplay.png``       -- canvas + Q-chart + dropdown row, V1 sp fixed vs PPO rescue.
* ``web_ladder.png``         -- ELO leaderboard panel after a fast-sim batch.
* ``web_replay.png``         -- replay mode mid-rally, including scrubber/frame info.
* ``web_qchart_detail.png``  -- tight crop of the per-action Q-value chart only.

Run from the repo root with the demo server already up::

    python -m http.server 8000 --directory web   # in another shell
    python scripts/capture_web_screenshots.py

The script uses Playwright sync API so it can be executed standalone (no
asyncio scaffolding required).  Browser is Chromium, headless, viewport
1400x950 to give the 1200px canvas and full mode-bar room to breathe.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
URL = os.environ.get("DEMO_URL", "http://localhost:8000/")
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "report" / "figures"
VIEWPORT = {"width": 1400, "height": 950}

# Pad around an element bounding box so we get a little breathing room.
PAD = 8


def _wait_for_loading_gone(page) -> None:
    """The #loading div picks up the .hidden class after agents are ready."""
    try:
        page.wait_for_selector("#loading.hidden", state="attached", timeout=20000)
    except PWTimeout:
        # Some flows (ladder/replay) hide the spinner immediately; fall back to
        # checking visibility instead of the class toggle.
        page.wait_for_selector("#loading", state="hidden", timeout=5000)


def _bbox_with_pad(page, selector: str) -> dict:
    """Return a clip rect spanning a list of selectors with PAD pixels of margin."""
    box = page.locator(selector).bounding_box()
    if box is None:
        raise RuntimeError(f"Could not locate {selector}")
    return {
        "x": max(0, box["x"] - PAD),
        "y": max(0, box["y"] - PAD),
        "width": box["width"] + 2 * PAD,
        "height": box["height"] + 2 * PAD,
    }


def _bbox_union(page, selectors: list[str]) -> dict:
    """Compute the minimal axis-aligned rectangle covering several selectors."""
    boxes = []
    for sel in selectors:
        b = page.locator(sel).bounding_box()
        if b is None:
            raise RuntimeError(f"Could not locate {sel}")
        boxes.append(b)
    x0 = min(b["x"] for b in boxes) - PAD
    y0 = min(b["y"] for b in boxes) - PAD
    x1 = max(b["x"] + b["width"] for b in boxes) + PAD
    y1 = max(b["y"] + b["height"] for b in boxes) + PAD
    return {
        "x": max(0, x0),
        "y": max(0, y0),
        "width": x1 - max(0, x0),
        "height": y1 - max(0, y0),
    }


def _select_option(page, selector: str, value: str) -> None:
    """Set a <select> by value and dispatch a change event so game.js rewires."""
    page.select_option(selector, value)
    # game.js listens to native change events, but make sure focus moves on so
    # any blur-triggered handlers also fire.
    page.evaluate("document.activeElement && document.activeElement.blur();")


# ----------------------------------------------------------------------------
# Capture helpers
# ----------------------------------------------------------------------------

def shot_freeplay(page) -> Path:
    page.goto(URL, wait_until="networkidle")
    _wait_for_loading_gone(page)

    _select_option(page, "#left-select", "v1_sp_fixed")
    _select_option(page, "#right-select", "ppo_rescue")
    # Restart so both new agents take effect immediately, then let the rally start.
    page.click("#restart")
    page.wait_for_timeout(3500)

    # Capture the canvas, the dropdown bar above it, and the Q-chart below.
    clip = _bbox_union(page, ["#mode-bar", "#game", "#qchart"])
    out = OUT_DIR / "web_freeplay.png"
    page.screenshot(path=str(out), clip=clip)
    return out


def shot_ladder(page) -> Path:
    page.goto(URL, wait_until="networkidle")
    _wait_for_loading_gone(page)

    page.click("input[name='mode'][value='ladder']")
    page.wait_for_selector("#ladder-panel", state="visible", timeout=5000)
    page.wait_for_timeout(400)  # let table render initial rows

    # Kick off 100 fast-sim matches and poll the progress span.  If it's slow
    # we accept partial completion (>= 30 matches) so we don't run over budget.
    page.click("#fast-sim-100")
    progress_locator = page.locator("#fast-sim-progress")
    deadline = time.time() + 90  # 90s hard cap
    matches_done = 0
    while time.time() < deadline:
        text = (progress_locator.inner_text() or "").strip()
        # Format from elo.js is e.g. "Running fast sim: 47/100" or empty when done.
        if text == "":
            # Either not started yet or finished — peek at the stop button to disambiguate.
            stop_visible = page.locator("#fast-sim-stop").is_visible()
            if not stop_visible and matches_done > 0:
                break
        else:
            try:
                matches_done = int(text.split("/")[0].split()[-1])
            except ValueError:
                matches_done = 0
            if matches_done >= 100:
                break
            if matches_done >= 30 and time.time() - deadline + 90 > 45:
                # 45s elapsed with at least 30 matches done -- good enough.
                break
        page.wait_for_timeout(500)

    # Make sure the table has settled, scroll panel into view, then capture.
    page.wait_for_timeout(500)
    page.evaluate("document.querySelector('#ladder-panel').scrollIntoView({block:'start'});")
    page.wait_for_timeout(200)

    clip = _bbox_with_pad(page, "#ladder-panel")
    out = OUT_DIR / "web_ladder.png"
    page.screenshot(path=str(out), clip=clip)
    return out, matches_done


def shot_replay(page) -> Path:
    page.goto(URL, wait_until="networkidle")
    _wait_for_loading_gone(page)

    page.click("input[name='mode'][value='replay']")
    page.wait_for_selector("#replay-panel", state="visible", timeout=5000)
    page.wait_for_timeout(400)

    # Pick the first concrete replay (skip the placeholder option).
    options = page.eval_on_selector_all(
        "#replay-select option",
        "els => els.map(e => ({value: e.value, label: e.textContent}))",
    )
    real = [o for o in options if o["value"]]
    if not real:
        raise RuntimeError("No replays available in #replay-select")
    _select_option(page, "#replay-select", real[0]["value"])
    page.wait_for_timeout(500)
    page.click("#replay-play")
    # Let the rally develop a bit so the canvas isn't a blank serve frame.
    page.wait_for_timeout(2200)

    # Grab the canvas plus the replay control row underneath.
    clip = _bbox_union(page, ["#game", "#replay-panel"])
    out = OUT_DIR / "web_replay.png"
    page.screenshot(path=str(out), clip=clip)
    return out, real[0]["label"]


def shot_qchart_detail(page) -> Path:
    page.goto(URL, wait_until="networkidle")
    _wait_for_loading_gone(page)

    _select_option(page, "#left-select", "v1_sp_fixed")
    _select_option(page, "#right-select", "rainbow_sp_fixed")
    page.click("#restart")
    # Need ~10s of rallies before the chart has populated columns of values.
    page.wait_for_timeout(10000)

    clip = _bbox_with_pad(page, "#qchart")
    out = OUT_DIR / "web_qchart_detail.png"
    page.screenshot(path=str(out), clip=clip)
    return out


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, Path, str]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(viewport=VIEWPORT, device_scale_factor=2)
            page = context.new_page()
            page.set_default_timeout(15000)

            print("[1/4] Free-play screenshot...", flush=True)
            path = shot_freeplay(page)
            results.append(("freeplay", path, "V1 sp fixed vs PPO rescue"))

            print("[2/4] Ladder screenshot...", flush=True)
            path, matches = shot_ladder(page)
            results.append(("ladder", path, f"{matches} matches simulated"))

            print("[3/4] Replay screenshot...", flush=True)
            path, label = shot_replay(page)
            results.append(("replay", path, f"replay='{label.strip()}'"))

            print("[4/4] Q-chart detail...", flush=True)
            path = shot_qchart_detail(page)
            results.append(("qchart", path, "V1 sp fixed vs Rainbow sp fixed"))
        finally:
            browser.close()

    print("\nSaved screenshots:")
    for name, path, note in results:
        size = path.stat().st_size if path.exists() else 0
        print(f"  {name:<8} -> {path} ({size//1024} KB)  [{note}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
