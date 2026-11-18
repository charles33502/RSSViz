"""檢查：靜置時「桌子對物體」的法向力 = 物體重量；夾持時兩指的力約相等。
用法： python test_rssdata.py [twins 資料夾]
"""
import sys
from pathlib import Path

import numpy as np

import rssdata as R

root = Path(sys.argv[1]) if len(sys.argv) > 1 else R.DEFAULT_TWINS
MASS = {"mallet": 0.45, "bin": 0.12, "cleaver": 0.35, "board": 0.9}  # 各分身 scene.py 裡寫的質量
fails = 0
for twin, files in R.find_runs(root).items():
    for f in files:
        run = R.load_run(f)
        print(f"{twin}/{f.name}: {len(run.t)} 步, {len(run.contacts)} 個接觸點, 有位置/法向: {run.has_geometry}")
        for obj in run.objects:
            if obj not in MASS:
                continue
            others = {b for b in run.body_names + [R.WORLD] if b != obj}
            s = R.force_series(run, {obj}, others)
            fz = float(s["Fz"].iloc[10:20].mean())  # 開頭靜置
            w = MASS[obj] * 9.81
            ok = abs(fz - w) < 0.03 * w
            fails += not ok
            print(f"  {'PASS' if ok else 'FAIL'} {obj}: 開頭靜置時其他東西對它的 Fz {fz:.2f} N，重量 {w:.2f} N")
        print("  曾接觸的對：", ", ".join(R.pair_summary(run)["對"].head(6)))
print("失敗數：", fails)
sys.exit(1 if fails else 0)
