"""RSSViz — robosimstudio 分身的物理記錄觀測

啟動：  ./run.sh [twins 資料夾] [port]
讀的是 physlog.py 寫的 run/<name>_phys.npz 與 run/<name>_contacts.jsonl。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import rssdata as R

st.set_page_config(page_title="RSSViz 分身物理觀測", page_icon="🤖", layout="wide")
PLOT_CFG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}


def cli_root() -> str:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--root", default=str(R.DEFAULT_TWINS))
    a, _ = p.parse_known_args(sys.argv[1:])
    return a.root


@st.cache_data(show_spinner="讀取記錄中…", max_entries=8)
def load(path: str, mtime: float) -> R.Run:
    return R.load_run(Path(path))


@st.cache_data(show_spinner=False, max_entries=8)
def load_shapes(path: str, mtime: float):
    return R.load_shapes(Path(path))


@st.cache_data(show_spinner="組合場景網格…", max_entries=64)
def meshes(path: str, mtime: float, k: int, mode: str):
    return R.scene_meshes(load_shapes(path, mtime), load(path, mtime), k, mode)


VIEWER = (Path(__file__).parent / "viewer.html").read_text(encoding="utf-8")


@st.cache_data(show_spinner="準備 3D 資料…", max_entries=16)
def viewer_html(path: str, mtime: float, mode: str, filt: tuple, hide_rest: bool, opacity: float, arrow: float,
                vid: str) -> str:
    run_ = load(path, mtime)
    c = run_.contacts
    if filt:
        S = {b for n in filt for b in run_.entities[n]}
        c = c[c["body_a"].isin(S) | c["body_b"].isin(S)]
    if hide_rest:
        ground = {"table", R.WORLD}
        c = c[~(c["body_a"].astype(str).map(R.short).isin(ground) & c["body_b"].astype(str).map(R.short).isin(ground))]
    every = 1 if len(run_.t) <= 1500 else int(np.ceil(len(run_.t) / 1500))
    data = R.viewer_payload(run_, load_shapes(path, mtime), mode, c, every)
    data["opt"] = {"opacity": opacity, "arrow": arrow, "start": 0}
    data["id"] = vid
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return VIEWER.replace("__DATA__", js)


GROUP_COLOR = {"左臂": "#d9d9d9", "右臂": "#d9d9d9", "左臂底座": "#9a9a9a", "右臂底座": "#9a9a9a",
               "左手指": "#5b5b66", "右手指": "#5b5b66", "立柱": "#3a3a3a", "table": "#2f5d6b"}
OBJ_COLORS = ["#e8a33d", "#3d6fe8", "#b04ad6", "#3dbb6a", "#e85d3d", "#8c6d46"]


def add_scene(fig, groups: dict, opacity: float, focus: set[str]):
    objs = [g for g in groups if g not in GROUP_COLOR]
    for g, (v, f) in groups.items():
        col = GROUP_COLOR.get(g) or OBJ_COLORS[objs.index(g) % len(OBJ_COLORS)]
        op = opacity * (0.35 if g == "table" else 1.0)
        if focus and g not in focus:
            op *= 0.4
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                color=col, opacity=op, name=g, flatshading=True, hoverinfo="name",
                                lighting=dict(ambient=0.55, diffuse=0.8, specular=0.2),
                                lightposition=dict(x=100, y=-200, z=300), showlegend=True,
                                legendgroup="scene", legendgrouptitle_text="場景"))


def show(fig, data: pd.DataFrame | None = None, name: str = "data"):
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
    if data is not None:
        with st.expander("📄 這張圖的數據"):
            st.dataframe(data, width="stretch", hide_index=True)
            st.download_button("下載 CSV", data.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{name}.csv", mime="text/csv", key=f"dl::{name}")


# ------------------------------------------------------------------ 來源
with st.sidebar:
    st.header("📂 記錄來源")
    root = Path(st.text_input("分身資料夾 (twins)", cli_root())).expanduser()
    runs = R.find_runs(root)
    if not runs:
        st.error("這裡找不到任何 run/*_phys.npz。\n\n在分身頁面按「routine (logged)」就會產生。")
        st.stop()
    twin = st.selectbox("分身", list(runs), key="twin")
    f = st.selectbox("記錄", runs[twin], format_func=lambda p: p.name)
    if st.button("🔄 重新讀取", width="stretch"):
        st.cache_data.clear()

run = load(str(f), f.stat().st_mtime)
dur = float(run.t[-1]) if len(run.t) else 0.0
st.title(f"🤖 {twin} · {run.name}")
c = st.columns(4)
c[0].metric("模擬時間", f"{dur:.2f} s")
c[1].metric("控制步數", f"{len(run.t):,}")
c[2].metric("接觸點紀錄", f"{len(run.contacts):,}")
c[3].metric("物體", ", ".join(run.objects) or "—")
if not run.has_geometry:
    st.warning("這份記錄是舊格式：沒有接觸點位置與法向，無法分出法向力 / 摩擦力，也沒有 3D 分布。"
               "在分身頁面重按一次「routine (logged)」即可更新。")

tabs = st.tabs(["🤝 接觸力", "📍 接觸點分布", "📦 物體運動", "🦾 關節", "✅ 物理檢查與錄影", "ℹ️ 說明"])
menu = run.entities
names = list(menu)

# ------------------------------------------------------------------ 接觸力
with tabs[0]:
    st.caption("選「誰」受「誰」的力。同一步裡兩組之間的所有接觸點會先加總，這才是有物理意義的力。")
    pairs = R.pair_summary(run)
    with st.expander("曾經接觸過的組合（點一下表格下方的按鈕可帶入）", expanded=False):
        st.dataframe(pairs.round(2), width="stretch", hide_index=True)
        if not pairs.empty:
            pick = st.selectbox("帶入哪一對", pairs["對"].tolist(), key="pairpick")
            if st.button("帶入"):
                a, b = pick.split(" ↔ ")
                st.session_state["A"], st.session_state["B"] = [a], [b]
    default_obj = run.objects[-1] if run.objects else names[0]
    st.session_state.setdefault("A", [default_obj])
    st.session_state.setdefault("B", ["右手指"] if "右手指" in menu else [n for n in names if n != default_obj][:1])
    c1, c2 = st.columns(2)
    A = c1.multiselect("受力方 A", names, key="A")
    B = c2.multiselect("施力方 B（對 A 施力）", names, key="B")
    o = st.columns(4)
    show_n = o[0].checkbox("法向力總和", True)
    show_t = o[1].checkbox("摩擦力", True)
    show_f = o[2].checkbox("合力大小", False)
    show_xyz = o[3].checkbox("合力三軸 Fx/Fy/Fz", False)
    m = st.number_input("畫一條「重量線」m·g（kg，0 = 不畫）", 0.0, 100.0, 0.0, 0.01)
    if A and B:
        SA = {b for n in A for b in menu[n]}
        SB = {b for n in B for b in menu[n]} - SA
        s = R.force_series(run, SA, SB)
        cols = ([c for c, on in (("法向力總和", show_n), ("摩擦力", show_t), ("合力", show_f)) if on]
                + (["Fx", "Fy", "Fz"] if show_xyz else []))
        if cols:
            long = s.melt(id_vars=["t"], value_vars=cols, var_name="量", value_name="N")
            fig = px.line(long, x="t", y="N", color="量", height=460,
                          title=f"{' + '.join(B)} 對 {' + '.join(A)} 的力")
            if m > 0:
                fig.add_hline(y=m * 9.81, line_dash="dot", annotation_text=f"m·g = {m * 9.81:.2f} N")
            fig.update_layout(xaxis_title="t [s]", yaxis_title="力 [N]", hovermode="x unified")
            show(fig, s, f"force_{'_'.join(A)}_by_{'_'.join(B)}")
        fig2 = px.area(s, x="t", y="接觸點數", height=200, title="接觸點數")
        fig2.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig2, width="stretch", config=PLOT_CFG)
        nz = s[s["接觸點數"] > 0]
        if not nz.empty and run.has_geometry:
            k = st.columns(3)
            k[0].metric("接觸期間", f"{nz.t.min():.2f} – {nz.t.max():.2f} s")
            k[1].metric("最大法向力總和", f"{nz['法向力總和'].max():.1f} N", f"t = {nz.loc[nz['法向力總和'].idxmax(), 't']:.2f} s", delta_color="off")
            k[2].metric("法向力總和中位數（接觸期間）", f"{nz['法向力總和'].median():.1f} N")

# ------------------------------------------------------------------ 接觸點分布
with tabs[1]:
    if not run.has_geometry:
        st.info("舊格式記錄沒有接觸點位置。")
    else:
        shp = load_shapes(str(f), f.stat().st_mtime)
        g1, g2, g3 = st.columns([2, 1, 1])
        if shp is None:
            g1.info("這份記錄沒有形狀檔（<name>_shapes.npz），只畫得出點。重按一次「routine (logged)」就會有。")
            smode = "不畫"
        else:
            smode = g1.radio("場景形狀", ["碰撞形狀", "外觀網格（較慢）", "不畫"], horizontal=True,
                             help="碰撞形狀是物理真正用來算接觸的；外觀網格是畫面上看到的機器人模型。")
        sop = g2.slider("場景不透明度", 0.05, 1.0, 0.5, 0.05)
        L = g3.slider("箭頭長度（每 N 幾公分，0 = 不畫）", 0.0, 2.0, 0.3, 0.05)
        h1, h2 = st.columns([1, 2])
        hide_rest = h1.checkbox("隱藏靜置在桌上 / 地上的接觸", value=True)
        filt = h2.multiselect("只看和這些有關的接觸（空白 = 全部）", names, key="pf")
        allc = run.contacts
        if filt:
            S = {b for n in filt for b in menu[n]}
            allc = allc[allc["body_a"].isin(S) | allc["body_b"].isin(S)]
        if hide_rest:
            ground = {"table", R.WORLD}
            allc = allc[~(allc["body_a"].astype(str).map(R.short).isin(ground)
                          & allc["body_b"].astype(str).map(R.short).isin(ground))]
        mode = {"碰撞形狀": "collision", "不畫": "none"}.get(smode, "visual")
        html = viewer_html(str(f), f.stat().st_mtime, mode, tuple(filt), hide_rest, sop, L / 100, f"{twin}/{run.name}")
        st.caption("時間軸、播放與視角都在下面的 3D 視窗裡操作（在瀏覽器裡跑，切換時間不會重設視角；視角也會記住）。"
                   "上方的選項改了才會重新載入。")
        components.html(html, height=780)
        with st.expander("📄 某一刻的接觸點（表格）"):
            ti = st.slider("時間 [s]", 0.0, dur, min(dur, 1.0), key="tbl_t")
            k = int(np.argmin(np.abs(run.t - ti)))
            d = allc[allc["step"] == run.step[k]].copy()
            d["對（受力方 ← 施力方）"] = d["body_b"].astype(str).map(R.body_group) + " ← " + d["body_a"].astype(str).map(R.body_group)
            d["點"] = d["b"].astype(str) + " ← " + d["a"].astype(str)
            st.dataframe(d[["點", "對（受力方 ← 施力方）", "fn", "ft", "fx", "fy", "fz", "px", "py", "pz", "nx", "ny", "nz", "depth"]]
                         .rename(columns={"fn": "法向力 N", "ft": "摩擦力 N", "fx": "Fx（作用在受力方）",
                                          "fy": "Fy", "fz": "Fz"}).round(4),
                         width="stretch", hide_index=True)

# ------------------------------------------------------------------ 物體運動
with tabs[2]:
    bodies = [b for b in run.body_names if not R.is_robot(b) and R.short(b) != "table"]
    show_robot = st.checkbox("也列出機器人的連桿", False)
    opts = run.body_names if show_robot else bodies
    pick = st.multiselect("物體", opts, default=bodies, format_func=R.short)
    qty = st.multiselect("量", ["位移 [mm]", "高度變化 [mm]", "速度 [m/s]", "角速度 [rad/s]", "轉角 [deg]", "x", "y", "z"],
                         default=["位移 [mm]", "高度變化 [mm]", "轉角 [deg]"])
    if pick and qty:
        frames = [R.body_motion(run, b).assign(物體=R.short(b)) for b in pick]
        mo = pd.concat(frames, ignore_index=True)
        long = mo.melt(id_vars=["t", "物體"], value_vars=qty, var_name="量", value_name="值")
        fig = px.line(long, x="t", y="值", color="物體", facet_row="量", height=230 * len(qty) + 60)
        fig.update_yaxes(matches=None, title_text="")
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=", 1)[-1]))
        fig.update_layout(hovermode="x unified")
        show(fig, mo, "motion")
        end = mo.groupby("物體").agg(最遠位移_mm=("位移 [mm]", "max"), 結束位移_mm=("位移 [mm]", "last"),
                                    最大轉角_deg=("轉角 [deg]", "max"), 最高抬升_mm=("高度變化 [mm]", "max"))
        st.dataframe(end.round(1), width="stretch")

# ------------------------------------------------------------------ 關節
with tabs[3]:
    jt = R.joint_table(run)
    arm = st.radio("手臂", ["left", "right"], horizontal=True, format_func=lambda s: "左臂" if s == "left" else "右臂")
    js = jt[jt.joint.str.startswith(arm + "_")]
    mode = st.radio("看什麼", ["實際角度 vs 目標", "追蹤誤差（實際 − 目標）"], horizontal=True)
    rows = []
    for j, ci in zip(js.joint, js.coord):
        rows.append(pd.DataFrame({"t": run.t, "關節": j, "實際": run.joint_q[:, ci], "目標": run.joint_target_q[:, ci]}))
    jd = pd.concat(rows, ignore_index=True)
    jd["誤差"] = jd["實際"] - jd["目標"]
    if mode.startswith("實際"):
        long = jd.melt(id_vars=["t", "關節"], value_vars=["實際", "目標"], var_name="", value_name="q")
        fig = px.line(long, x="t", y="q", color="", line_dash="", facet_row="關節", height=170 * len(js) + 60)
    else:
        fig = px.line(jd, x="t", y="誤差", facet_row="關節", height=170 * len(js) + 60)
    fig.update_yaxes(matches=None, title_text="")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=", 1)[-1]))
    st.caption("單位：轉動關節 rad，手指 m（開口）。手指的誤差在夾住東西時本來就不會是 0：目標是「全合」，被物體擋住。")
    show(fig, jd, f"joints_{arm}")

# ------------------------------------------------------------------ 物理檢查與錄影
with tabs[4]:
    ph = R.load_physics(run.folder)
    if ph:
        st.subheader(("✅ 通過" if ph.get("passed") else "❌ 未通過") + "  ·  ./ctl physics")
        st.dataframe(pd.DataFrame(ph["checks"]).rename(columns={"ok": "通過", "name": "項目", "detail": "說明"}),
                     width="stretch", hide_index=True)
    else:
        st.info("這個分身還沒跑過 ./ctl physics。")
    vids = R.takes(run.folder)
    if vids:
        v = st.selectbox("錄影（./ctl play --record）", vids, format_func=lambda p: p.name)
        st.video(str(v))
        st.caption("注意：錄影和這份物理記錄是不同次的執行；物理是確定性的，結果應該一致。")

# ------------------------------------------------------------------ 說明
with tabs[5]:
    st.markdown("""
**資料**：`run/<name>_phys.npz`（每一步的剛體位姿、速度、關節）與 `run/<name>_contacts.jsonl`
（每一步的每一個接觸點），由分身的 `physlog.py`（頁面上的「routine (logged)」）寫出。

**力的定義**
- 每個接觸點的 `force` 是作用在第二個形狀 `b` 上的力 [N，世界座標]；本工具在「A 受 B 的力」裡自動處理正負號。
- **法向力**：沿接觸法向、互相壓的分量，永遠 ≥ 0。
- **摩擦力**：扣掉法向後剩下的切向分量（同一步先向量相加再取大小）。
- 同一對物體在同一步通常有好幾個接觸點；**單點的力分配不唯一、會跳動**，請看加總。
- 記錄的是每個控制步（0.02 s）最後一個物理子步（0.002 s）的值，不是平均。

**驗證**：靜置時「其他所有東西對某物體的 Fz」= 該物體重量（`python test_rssdata.py`）。
""")
