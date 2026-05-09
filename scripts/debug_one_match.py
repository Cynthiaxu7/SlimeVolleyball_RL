"""Run ONE match in the actual browser worker (V1 selfplay vs Baseline) and
print the action histogram + Q values. Confirms whether the worker is
producing sensible inferences."""
import asyncio
from playwright.async_api import async_playwright

JS = """async ({left, right, lurl, rurl}) => {
  return await new Promise((resolve, reject) => {
    const w = new Worker('match_worker.js');
    w.onmessage = (e) => { w.terminate(); resolve(e.data); };
    w.onerror = (e) => { w.terminate(); reject(String(e)); };
    w.postMessage({
      leftVariantKey: left, rightVariantKey: right,
      leftModelUrl: lurl, rightModelUrl: rurl,
      seed: 42, maxSteps: 3100,
    });
  });
}"""

async def run_one(left, right, lurl, rurl):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("http://127.0.0.1:18765/")
        await page.wait_for_load_state("networkidle", timeout=10_000)
        result = await page.evaluate(JS, {"left": left, "right": right, "lurl": lurl, "rurl": rurl})
        await browser.close()
        return result

async def main():
    cases = [
        ("v1_selfplay vs Baseline (right=trained)",
         "baseline", "v1_selfplay", None, "model_v1_selfplay.onnx"),
        ("Baseline vs v1_selfplay (left=trained)",
         "v1_selfplay", "baseline", "model_v1_selfplay.onnx", None),
    ]
    for label, l, r, lurl, rurl in cases:
        print(f"\n=== {label} ===")
        res = await run_one(l, r, lurl, rurl)
        d = res.get("debug", {})
        print(f"  outcome: leftLives={res['leftLives']}  rightLives={res['rightLives']}  steps={res['steps']}")
        labels = ['NOOP', 'LEFT', 'L+J', 'JUMP', 'R+J', 'RIGHT']
        if d.get("lActHist"):
            tot = sum(d["lActHist"]) or 1
            print(f"  left  ({d.get('leftVariant')}) action distribution:")
            for i, c in enumerate(d["lActHist"]):
                print(f"    {labels[i]:5s} {c:5d} ({100*c/tot:5.1f}%)")
        if d.get("rActHist"):
            tot = sum(d["rActHist"]) or 1
            print(f"  right ({d.get('rightVariant')}) action distribution:")
            for i, c in enumerate(d["rActHist"]):
                print(f"    {labels[i]:5s} {c:5d} ({100*c/tot:5.1f}%)")
        if d.get("firstQL") is not None:
            print(f"  step-50 left obs:  {[round(x,3) for x in d['firstObsL']]}")
            print(f"  step-50 left  Q:   {[round(x,4) for x in d['firstQL']]}")
        if d.get("firstQR") is not None:
            print(f"  step-50 right obs: {[round(x,3) for x in d['firstObsR']]}")
            print(f"  step-50 right Q:   {[round(x,4) for x in d['firstQR']]}")

asyncio.run(main())
