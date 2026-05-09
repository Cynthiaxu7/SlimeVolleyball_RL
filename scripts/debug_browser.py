"""Headless chromium driving the ladder fast-sim, capturing console output
and the final ELO table. Compares to the Python ground truth from ladder_sim.

Run: python scripts/debug_browser.py --n-matches 100
"""
import argparse, asyncio, json, sys, time
from playwright.async_api import async_playwright


async def main(n_matches: int, port: int):
    url = f"http://127.0.0.1:{port}/"
    console_lines: list[str] = []
    page_errors: list[str] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context()
        page = await ctx.new_page()

        page.on("console", lambda msg: console_lines.append(f"[{msg.type}] {msg.text}"))
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))

        await page.goto(url)
        await page.wait_for_load_state("networkidle", timeout=10_000)

        # 1) Switch to Ladder mode
        await page.click('input[name="mode"][value="ladder"]')
        await asyncio.sleep(0.5)

        # 2) Reset ELO so we start fresh
        reset_btn = page.locator("#reset-elo")
        if await reset_btn.count():
            await reset_btn.click()
            await asyncio.sleep(0.5)

        # 3) Click "Run 100 matches" or use JS to set custom target
        # We'll inject directly: window-level call to runFastSim if exposed,
        # or click the Run-100 button n_matches/100 times.
        # Easiest: just click Run 1000 once and wait.
        if n_matches >= 1000:
            await page.click("#fast-sim-1000")
        elif n_matches >= 100:
            await page.click("#fast-sim-100")
        else:
            # Fallback: directly invoke via JS with target=n_matches
            await page.evaluate(f"window.__fastsim = (typeof runFastSim === 'function') ? runFastSim({n_matches}) : null;")

        # 4) Poll progress every 2s until status text shows completion
        start = time.time()
        last_progress = ""
        while True:
            txt = (await page.locator("#fast-sim-progress").text_content()) or ""
            if txt != last_progress:
                print(f"[browser progress] {txt}")
                last_progress = txt
            # Heuristic: progress text empty after fast-sim ends
            if (txt == "" or "completed" in txt.lower() or "done" in txt.lower()) and (time.time() - start > 5):
                # also check ladder-status
                status = (await page.locator("#ladder-status").text_content()) or ""
                if "completed" in status.lower() or txt == "":
                    break
            if time.time() - start > 600:
                print("[debug] timeout waiting for fast-sim")
                break
            await asyncio.sleep(2)

        # 5) Read ELO table
        rows = await page.eval_on_selector_all(
            "#ladder-table tbody tr",
            """rs => rs.map(r => Array.from(r.children).map(td => td.textContent.trim()))""",
        )
        print("\n=== Browser ladder table after fast-sim ===")
        print("| # | Player | ELO | Games | W-D-L |")
        print("|---|--------|-----|-------|-------|")
        for r in rows:
            print("| " + " | ".join(r) + " |")

        # 6) Print console + errors
        print("\n=== Console messages (last 30) ===")
        for line in console_lines[-30:]:
            print(line)

        print(f"\n=== Page errors ({len(page_errors)}) ===")
        for e in page_errors:
            print(e)

        # 7) Read the actual ladder roster JS object for raw numbers
        roster_js = await page.evaluate("""() => {
            if (typeof ladder === 'undefined') return null;
            return ladder.roster.map(r => ({
                id: r.id, name: r.name, variantKey: r.variantKey,
                eloRating: r.eloRating, gamesPlayed: r.gamesPlayed,
                wins: r.wins, draws: r.draws, losses: r.losses,
                unavailable: r.unavailable,
            }));
        }""")
        if roster_js:
            print("\n=== Roster JSON (sorted by ELO desc) ===")
            for r in sorted(roster_js, key=lambda r: -r["eloRating"]):
                print(f"  {r['name']:18s}  ELO={r['eloRating']:7.1f}  "
                      f"games={r['gamesPlayed']:4d}  W-D-L={r['wins']}-{r['draws']}-{r['losses']}  "
                      f"unavail={r['unavailable']}")

        await browser.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-matches", type=int, default=100)
    ap.add_argument("--port", type=int, default=18765)
    args = ap.parse_args()
    asyncio.run(main(args.n_matches, args.port))
