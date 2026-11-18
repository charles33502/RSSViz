"""讀 robosimstudio live 分身的記錄（physlog.py 寫的 <name>_phys.npz 與 <name>_contacts.jsonl）。

資料單位：長度 m、時間 s、力 N（世界座標，z 朝上）。
接觸的力向量 ``force`` 作用在每一筆的第二個形狀 ``b`` 上；``normal`` 由 ``a`` 指向 ``b``，
所以 ``normal_force`` 不管 a、b 誰先誰後，正值都代表「互相壓」。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TWINS = Path.home() / "robosimstudio/examples/live/twins"
WORLD = "(世界/地面)"


# ------------------------------------------------------------------ 名稱與分組
def short(name: str) -> str:
    return str(name).replace("\\", "/").rstrip("/").split("/")[-1]


def is_robot(full: str) -> bool:
    return "/worldbody/" in str(full)


def arm_of(full: str) -> str | None:
    s = short(full)
    for side in ("left", "right"):
        if s.startswith(side + "_"):
            return side
    return None


SIDE = {"left": "左", "right": "右"}


def body_group(full: str) -> str:
    """一個剛體歸到哪一組：左手指 / 右手指 / 左臂 / 右臂 / 物體自己的名字。"""
    if full == WORLD:
        return WORLD
    if is_robot(full):
        side = arm_of(full) or "?"
        s = short(full)
        return f"{SIDE.get(side, side)}手指" if "finger" in s else f"{SIDE.get(side, side)}臂"
    return short(full)


def finger_label(full: str) -> str:
    """手指再細分：左手-左指 / 左手-右指。"""
    s = short(full)
    side = arm_of(full) or "?"
    which = "左指" if "left_finger" in s else "右指" if "right_finger" in s else s
    return f"{SIDE.get(side, side)}手-{which}"


# ------------------------------------------------------------------ 資料
@dataclass
class Run:
    name: str
    folder: Path
    t: np.ndarray
    step: np.ndarray
    body_names: list[str]
    body_q: np.ndarray
    body_qd: np.ndarray
    joint_names: list[str]
    joint_q: np.ndarray
    joint_qd: np.ndarray
    joint_target_q: np.ndarray
    contacts: pd.DataFrame
    has_geometry: bool
    entities: dict[str, list[str]] = field(default_factory=dict)  # 選單上的名字 -> 剛體全名

    @property
    def objects(self) -> list[str]:
        return [b for b in self.body_names if not is_robot(b)]


def find_runs(twins_root: Path) -> dict[str, list[Path]]:
    """{分身名稱: [<name>_phys.npz, ...]}"""
    out: dict[str, list[Path]] = {}
    if not twins_root.is_dir():
        return out
    for twin in sorted(p for p in twins_root.iterdir() if p.is_dir()):
        files = sorted((twin / "run").glob("*_phys.npz"))
        if files:
            out[twin.name] = files
    return out


def _shape_tables(z):
    shape_names = [str(s) for s in z["shape_names"]]
    shape_body = np.asarray(z["shape_body"]).astype(int)
    body_names = [str(b) for b in z["body_names"]]
    label_to_body: dict[str, str] = {}
    for i, lab in enumerate(shape_names):
        b = shape_body[i] if i < len(shape_body) else -1
        label_to_body[lab] = body_names[b] if 0 <= b < len(body_names) else WORLD
    return shape_names, label_to_body


def load_contacts(path: Path, shape_names, label_to_body) -> tuple[pd.DataFrame, bool]:
    cols = {k: [] for k in ("step", "t", "a", "b", "body_a", "body_b",
                            "fx", "fy", "fz", "px", "py", "pz", "nx", "ny", "nz", "fn", "ft", "depth")}
    geometry = False
    if not path.exists():
        return pd.DataFrame(cols), False

    def body(lab):
        if isinstance(lab, int):
            lab = shape_names[lab] if 0 <= lab < len(shape_names) else str(lab)
        return label_to_body.get(lab, WORLD)

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            for p in d["pairs"]:
                force = p.get("force")
                if force is None:  # 舊格式：只有 impulse，除以子步長 0.002 s
                    force = [v / 0.002 for v in p["impulse"]]
                cols["step"].append(d["step"]); cols["t"].append(d["t"])
                cols["a"].append(short(p["a"])); cols["b"].append(short(p["b"]))
                cols["body_a"].append(body(p["a"])); cols["body_b"].append(body(p["b"]))
                cols["fx"].append(force[0]); cols["fy"].append(force[1]); cols["fz"].append(force[2])
                cols["depth"].append(p.get("depth", np.nan))
                if "pos" in p:
                    geometry = True
                    pos, nrm = p["pos"], p["normal"]
                    cols["px"].append(pos[0]); cols["py"].append(pos[1]); cols["pz"].append(pos[2])
                    cols["nx"].append(nrm[0]); cols["ny"].append(nrm[1]); cols["nz"].append(nrm[2])
                    cols["fn"].append(p["normal_force"]); cols["ft"].append(p["friction_force"])
                else:
                    for k in ("px", "py", "pz", "nx", "ny", "nz", "fn", "ft"):
                        cols[k].append(np.nan)
    df = pd.DataFrame(cols)
    for k in ("body_a", "body_b", "a", "b"):
        df[k] = df[k].astype("category")
    return df, geometry


def load_run(npz_path: Path) -> Run:
    npz_path = Path(npz_path)
    z = np.load(npz_path, allow_pickle=False)
    name = npz_path.name[: -len("_phys.npz")]
    shape_names, label_to_body = _shape_tables(z)
    contacts, geometry = load_contacts(npz_path.with_name(f"{name}_contacts.jsonl"), shape_names, label_to_body)
    body_names = [str(b) for b in z["body_names"]]
    run = Run(
        name=name, folder=npz_path.parent.parent,
        t=np.asarray(z["t"], float), step=np.asarray(z["step"]),
        body_names=body_names, body_q=np.asarray(z["body_q"], float), body_qd=np.asarray(z["body_qd"], float),
        joint_names=[str(j) for j in z["joint_names"]],
        joint_q=np.asarray(z["joint_q"], float), joint_qd=np.asarray(z["joint_qd"], float),
        joint_target_q=np.asarray(z["joint_target_q"], float),
        contacts=contacts, has_geometry=geometry,
    )
    run.entities = entity_menu(body_names + [WORLD])
    return run


def entity_menu(bodies: list[str]) -> dict[str, list[str]]:
    """選單：手指與手臂的分組、每個物體、每根手指、每節連桿。"""
    menu: dict[str, list[str]] = {}
    for b in bodies:
        menu.setdefault(body_group(b), []).append(b)
    for b in bodies:
        if is_robot(b) and "finger" in short(b):
            menu.setdefault(finger_label(b), []).append(b)
    for b in bodies:
        if is_robot(b) and "finger" not in short(b):
            menu.setdefault(short(b), []).append(b)
    return menu


# ------------------------------------------------------------------ 計算
def between(df: pd.DataFrame, A: set[str], B: set[str]) -> pd.DataFrame:
    """A、B 兩組剛體之間的每一個接觸點；加上「B 對 A 施的力」(fax, fay, faz)。"""
    ab = df["body_a"].isin(A) & df["body_b"].isin(B)
    ba = df["body_a"].isin(B) & df["body_b"].isin(A)
    sub = df[ab | ba].copy()
    sign = np.where(ab[ab | ba].to_numpy(), -1.0, 1.0)  # force 作用在 b：a∈A 時 A 受的是 -force
    sub["fax"], sub["fay"], sub["faz"] = sub["fx"] * sign, sub["fy"] * sign, sub["fz"] * sign
    return sub


def force_series(run: Run, A: set[str], B: set[str]) -> pd.DataFrame:
    """每一步：B 對 A 的法向力總和、摩擦力（向量和的大小）、合力的三軸與大小、接觸點數。"""
    sub = between(run.contacts, A, B)
    base = pd.DataFrame({"step": run.step, "t": run.t})
    if sub.empty:
        for k in ("法向力總和", "摩擦力", "合力", "Fx", "Fy", "Fz", "接觸點數"):
            base[k] = 0.0
        return base
    # 切向分量（向量）：A 受力扣掉沿法向的部分
    if run.has_geometry:
        n = sub[["nx", "ny", "nz"]].to_numpy()
        fa = sub[["fax", "fay", "faz"]].to_numpy()
        tang = fa - (fa * n).sum(1, keepdims=True) * n
        sub["tx"], sub["ty"], sub["tz"] = tang[:, 0], tang[:, 1], tang[:, 2]
    else:
        sub["tx"] = sub["ty"] = sub["tz"] = np.nan
    g = sub.groupby("step", observed=True).agg(
        法向力總和=("fn", "sum"), Fx=("fax", "sum"), Fy=("fay", "sum"), Fz=("faz", "sum"),
        tx=("tx", "sum"), ty=("ty", "sum"), tz=("tz", "sum"), 接觸點數=("fx", "size"))
    g["摩擦力"] = np.sqrt(g.tx ** 2 + g.ty ** 2 + g.tz ** 2)
    g["合力"] = np.sqrt(g.Fx ** 2 + g.Fy ** 2 + g.Fz ** 2)
    if not run.has_geometry:
        g["法向力總和"] = np.nan
        g["摩擦力"] = np.nan
    out = base.merge(g.drop(columns=["tx", "ty", "tz"]).reset_index(), on="step", how="left")
    return out.fillna({"接觸點數": 0, "Fx": 0, "Fy": 0, "Fz": 0, "合力": 0,
                       **({"法向力總和": 0, "摩擦力": 0} if run.has_geometry else {})})


def pair_summary(run: Run) -> pd.DataFrame:
    """曾經接觸過的每一對（分組層級）：接觸幾步、最大與平均法向壓力（同一步的點先加總）。"""
    df = run.contacts
    if df.empty:
        return pd.DataFrame()
    ga = df["body_a"].map(lambda b: body_group(b)).astype(str)
    gb = df["body_b"].map(lambda b: body_group(b)).astype(str)
    key = np.where(ga <= gb, ga + " ↔ " + gb, gb + " ↔ " + ga)
    d = pd.DataFrame({"對": key, "step": df["step"], "fn": df["fn"],
                      "f": np.sqrt(df.fx ** 2 + df.fy ** 2 + df.fz ** 2)})
    per = d.groupby(["對", "step"]).agg(fn=("fn", "sum"), f=("f", "sum"), n=("f", "size")).reset_index()
    s = per.groupby("對").agg(接觸步數=("step", "nunique"), 平均接觸點數=("n", "mean"),
                             最大法向力總和=("fn", "max"), 平均法向力總和=("fn", "mean"))
    return s.sort_values("最大法向力總和", ascending=False).reset_index()


def quat_angle_deg(q0: np.ndarray, q: np.ndarray) -> np.ndarray:
    """兩個四元數 (xyzw) 之間的旋轉角 [deg]。"""
    dot = np.abs((q0[None, :] * q).sum(-1))
    return np.degrees(2 * np.arccos(np.clip(dot, -1, 1)))


def body_motion(run: Run, body: str) -> pd.DataFrame:
    i = run.body_names.index(body)
    p = run.body_q[:, i, :3]
    q = run.body_q[:, i, 3:7]
    v = run.body_qd[:, i, :3]
    w = run.body_qd[:, i, 3:6]
    return pd.DataFrame({
        "t": run.t, "x": p[:, 0], "y": p[:, 1], "z": p[:, 2],
        "位移 [mm]": np.linalg.norm(p - p[0], axis=1) * 1e3,
        "高度變化 [mm]": (p[:, 2] - p[0, 2]) * 1e3,
        "速度 [m/s]": np.linalg.norm(v, axis=1),
        "角速度 [rad/s]": np.linalg.norm(w, axis=1),
        "轉角 [deg]": quat_angle_deg(q[0], q),
    })


def joint_table(run: Run) -> pd.DataFrame:
    """手臂與手指的關節（物體的自由關節略過）。joint_q 與 joint_target_q 前 16 個座標是兩手臂各 6+2。"""
    rows = []
    names = [short(j) for j in run.joint_names]
    k = 0
    for j, full in zip(names, run.joint_names):
        if not is_robot(full):
            break
        rows.append((j, k))
        k += 1
    return pd.DataFrame(rows, columns=["joint", "coord"])


def load_physics(folder: Path) -> dict | None:
    p = folder / "run" / "physics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def takes(folder: Path) -> list[Path]:
    return sorted((folder / "run").glob("*.mp4"), key=os.path.getmtime, reverse=True)


# ------------------------------------------------------------------ 形狀（<name>_shapes.npz）
GEO = {"PLANE": 1, "HFIELD": 2, "SPHERE": 3, "CAPSULE": 4, "ELLIPSOID": 5, "CYLINDER": 6,
       "BOX": 7, "MESH": 8, "CONE": 9, "CONVEX_MESH": 10}
VISIBLE, COLLIDES = 1, 2


def load_shapes(npz_path: Path) -> dict | None:
    p = Path(npz_path)
    p = p.with_name(p.name.replace("_phys.npz", "_shapes.npz"))
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _rotmat(q):  # xyzw
    x, y, zq, w = q
    return np.array([[1 - 2 * (y * y + zq * zq), 2 * (x * y - zq * w), 2 * (x * zq + y * w)],
                     [2 * (x * y + zq * w), 1 - 2 * (x * x + zq * zq), 2 * (y * zq - x * w)],
                     [2 * (x * zq - y * w), 2 * (y * zq + x * w), 1 - 2 * (x * x + y * y)]])


def _box(h):
    hx, hy, hz = h
    v = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, f


def _revolve(profile, n=20):
    """profile: [(r, z)] from bottom to top; a closed surface of revolution about z."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v = [[r * np.cos(t), r * np.sin(t), z] for r, z in profile for t in a]
    f = []
    for i in range(len(profile) - 1):
        for j in range(n):
            p, q = i * n + j, i * n + (j + 1) % n
            f += [[p, q, q + n], [p, q + n, p + n]]
    v = np.array(v, float)
    # caps
    base = len(v)
    v = np.vstack([v, [0, 0, profile[0][1]], [0, 0, profile[-1][1]]])
    top = (len(profile) - 1) * n
    for j in range(n):
        f += [[base, (j + 1) % n, j], [base + 1, top + j, top + (j + 1) % n]]
    return v, np.array(f)


def _primitive(t, s):
    if t == GEO["BOX"]:
        return _box(s)
    if t == GEO["SPHERE"]:
        r = s[0]
        prof = [(r * np.sin(a), -r * np.cos(a)) for a in np.linspace(0.01, np.pi - 0.01, 9)]
        return _revolve(prof, 14)
    if t == GEO["CYLINDER"]:
        r, h = s[0], s[1]
        return _revolve([(r, -h), (r, h)], 28)
    if t == GEO["CAPSULE"]:
        r, h = s[0], s[1]
        lo = [(r * np.sin(a), -h - r * np.cos(a)) for a in np.linspace(0.01, np.pi / 2, 5)]
        hi = [(r * np.cos(a), h + r * np.sin(a)) for a in np.linspace(0, np.pi / 2 - 0.01, 5)]
        return _revolve(lo + hi, 14)
    if t == GEO["CONE"]:
        r, h = s[0], s[1]
        return _revolve([(r, -h), (1e-4, h)], 20)
    return None


def scene_meshes(shapes: dict, run: Run, k: int, mode: str = "visual") -> dict[str, tuple]:
    """第 k 步的場景，依組（左臂、右手指、mallet…）合成一個網格：{組: (vertices, faces)}。
    mode: "visual" 畫外觀網格，"collision" 畫碰撞形狀。"""
    out: dict[str, list] = {}
    names = [str(s) for s in shapes["shape_names"]]
    vs, fs = shapes["mesh_vstart"], shapes["mesh_fstart"]
    for i, lab in enumerate(names):
        t = int(shapes["shape_type"][i])
        flags = int(shapes["shape_flags"][i])
        if t in (GEO["PLANE"], GEO["HFIELD"]) or flags & 8:
            continue
        if mode == "visual" and not flags & VISIBLE:
            continue
        if mode == "collision" and not flags & COLLIDES:
            continue
        s = shapes["shape_scale"][i]
        if t in (GEO["MESH"], GEO["CONVEX_MESH"]):
            v = shapes["mesh_vertices"][vs[i]:vs[i + 1]] * s
            f = shapes["mesh_faces"][fs[i]:fs[i + 1]]
            if len(v) == 0:
                continue
        else:
            prim = _primitive(t, s)
            if prim is None:
                continue
            v, f = prim
        tr = shapes["shape_transform"][i]
        v = v @ _rotmat(tr[3:7]).T + tr[:3]
        b = int(shapes["shape_body"][i])
        if 0 <= b < len(run.body_names):
            q = run.body_q[k, b]
            v = v @ _rotmat(q[3:7]).T + q[:3]
            group = body_group(run.body_names[b])
        else:
            side = "左" if "left_" in lab else "右" if "right_" in lab else ""
            group = f"{side}臂底座" if "_arm_geom" in lab else ("立柱" if lab.startswith("post") else short(lab))
        out.setdefault(group, []).append((v, f))
    merged = {}
    for g, parts in out.items():
        off, V, F = 0, [], []
        for v, f in parts:
            V.append(v)
            F.append(f + off)
            off += len(v)
        merged[g] = (np.vstack(V), np.vstack(F))
    return merged


# ------------------------------------------------------------------ 3D 檢視器的資料（一次傳給瀏覽器）
def viewer_payload(run: Run, shapes: dict | None, mode: str, contacts: pd.DataFrame, every: int = 1) -> dict:
    """形狀（各剛體自己的座標系）、每一步的剛體位姿、每一步的接觸點。瀏覽器端依時間組合，不必回伺服器。"""
    steps = np.arange(0, len(run.t), max(1, every))
    # --- 形狀：同一個剛體、同一組的三角形合併成一塊
    parts: dict[tuple[int, str], list] = {}
    if shapes is not None and mode != "none":
        names = [str(s) for s in shapes["shape_names"]]
        vs, fs = shapes["mesh_vstart"], shapes["mesh_fstart"]
        for i, lab in enumerate(names):
            t = int(shapes["shape_type"][i])
            flags = int(shapes["shape_flags"][i])
            if t in (GEO["PLANE"], GEO["HFIELD"]) or flags & 8:
                continue
            if (mode == "visual" and not flags & VISIBLE) or (mode == "collision" and not flags & COLLIDES):
                continue
            sc = shapes["shape_scale"][i]
            if t in (GEO["MESH"], GEO["CONVEX_MESH"]):
                v = shapes["mesh_vertices"][vs[i]:vs[i + 1]] * sc
                f = shapes["mesh_faces"][fs[i]:fs[i + 1]]
                if not len(v):
                    continue
            else:
                prim = _primitive(t, sc)
                if prim is None:
                    continue
                v, f = prim
            tr = shapes["shape_transform"][i]
            v = v @ _rotmat(tr[3:7]).T + tr[:3]
            b = int(shapes["shape_body"][i])
            if 0 <= b < len(run.body_names):
                group = body_group(run.body_names[b])
            else:
                b = -1
                side = "左" if "left_" in lab else "右" if "right_" in lab else ""
                group = f"{side}臂底座" if "_arm_geom" in lab else ("立柱" if lab.startswith("post") else short(lab))
            parts.setdefault((b, group), []).append((v, f))
    meshes = []
    for (b, group), lst in parts.items():
        off, V, F = 0, [], []
        for v, f in lst:
            V.append(v)
            F.append(f + off)
            off += len(v)
        V, F = np.vstack(V), np.vstack(F)
        meshes.append({"body": b, "group": group,
                       "v": np.round(V.ravel(), 5).tolist(), "f": F.ravel().astype(int).tolist()})
    used = sorted({m["body"] for m in meshes if m["body"] >= 0})
    poses = {str(b): np.round(run.body_q[steps, b, :7], 5).ravel().tolist() for b in used}
    # --- 接觸點：依步排好，offsets[k] .. offsets[k+1]
    c = contacts[contacts["step"].isin(run.step[steps])]
    step_to_k = {int(s): k for k, s in enumerate(run.step[steps])}
    c = c.assign(_k=c["step"].map(step_to_k)).sort_values("_k", kind="stable")
    pair = (c["body_b"].astype(str).map(body_group) + " ← " + c["body_a"].astype(str).map(body_group))
    pair_names = sorted(pair.unique().tolist())
    pair_idx = pair.map({p: i for i, p in enumerate(pair_names)}).to_numpy()
    counts = np.bincount(c["_k"].to_numpy(), minlength=len(steps))
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(int).tolist()
    point_names = (c["b"].astype(str) + " ← " + c["a"].astype(str))
    pn_names = sorted(point_names.unique().tolist())
    pn_idx = point_names.map({p: i for i, p in enumerate(pn_names)}).to_numpy()
    fn = c["fn"].clip(lower=0).to_numpy()
    cmax = float(np.quantile(fn, 0.99)) if len(fn) else 1.0
    bbox = None
    if len(c):
        p = c[["px", "py", "pz"]].to_numpy()
        bbox = [np.nanpercentile(p, 2, axis=0).round(4).tolist(), np.nanpercentile(p, 98, axis=0).round(4).tolist()]
    return {
        "t": np.round(run.t[steps], 4).tolist(),
        "meshes": meshes,
        "poses": poses,
        "contacts": {
            "offsets": offsets,
            "pos": np.round(c[["px", "py", "pz"]].to_numpy().ravel(), 5).tolist(),
            "nrm": np.round(c[["nx", "ny", "nz"]].to_numpy().ravel(), 4).tolist(),
            "fn": np.round(fn, 3).tolist(),
            "ft": np.round(c["ft"].to_numpy(), 3).tolist(),
            "depth": np.round(c["depth"].to_numpy() * 1e3, 4).tolist(),
            "pair": pair_idx.astype(int).tolist(), "pairs": pair_names,
            "point": pn_idx.astype(int).tolist(), "points": pn_names,
        },
        "cmax": max(cmax, 1e-6),
        "focus": bbox,
    }
