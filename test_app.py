"""Headless：每個分身的每份記錄、每個分頁跑一次，確認沒有例外。"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

import rssdata as R

here = Path(__file__).parent
fails = 0
for twin, files in R.find_runs(R.DEFAULT_TWINS).items():
    for f in files:
        at = AppTest.from_file(str(here / "app.py"), default_timeout=120)
        at.run()
        at.selectbox(key="twin").set_value(twin).run()
        if at.exception:
            fails += 1
            print("FAIL", twin, at.exception[0].value)
        else:
            print("ok  ", twin, f.name, f"{len(at.tabs)} 個分頁")
print("失敗數：", fails)
raise SystemExit(1 if fails else 0)
