# SPDX-License-Identifier: Apache-2.0
"""Every control step of a routine, written out: poses, velocities, joints and contacts.

``logged(routine)`` wraps a routine; it writes ``run/<name>_shapes.npz`` (every shape's
geometry, once, to draw the scene from the poses), and after each step it reads the sim's state and,
when the routine ends, writes ``run/<name>_phys.npz`` (arrays, one row a step) and
``run/<name>_contacts.jsonl`` (one line a step: every contact point -- the two shapes,
the force on the second [N, world], where it is, its normal, and that force split
into normal and friction parts).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parent / "run"


def _labels(arr) -> list[str]:
    return [str(x) for x in (arr or [])]


def _write_shapes(model, path: Path) -> None:
    """Every shape once: its body, type, size, pose on the body, flags, colour, and (meshes) its triangles.

    Enough to draw the scene at any step from ``body_q``. Types are newton.GeoType's numbers;
    ``scale`` is a box's half sizes, a sphere's radius, a capsule's or cylinder's (radius, half
    height) along its z, a mesh's scale; flags bit 1 is drawn, bit 2 collides.
    """
    n = int(model.shape_count)
    verts, faces, vstart, fstart = [], [], [0], [0]
    for src in model.shape_source:
        v = np.zeros((0, 3), np.float32)
        f = np.zeros((0, 3), np.int32)
        if src is not None and getattr(src, "vertices", None) is not None:
            v = np.asarray(src.vertices, np.float32).reshape(-1, 3)
            f = np.asarray(src.indices, np.int32).reshape(-1, 3)
        verts.append(v)
        faces.append(f)
        vstart.append(vstart[-1] + len(v))
        fstart.append(fstart[-1] + len(f))
    color = getattr(model, "shape_color", None)
    np.savez_compressed(
        path,
        shape_names=np.array(_labels(getattr(model, "shape_label", None))),
        shape_body=np.asarray(model.shape_body.numpy()),
        shape_type=np.asarray(model.shape_type.numpy()),
        shape_scale=np.asarray(model.shape_scale.numpy(), np.float32),
        shape_transform=np.asarray(model.shape_transform.numpy(), np.float32),
        shape_flags=np.asarray(model.shape_flags.numpy()),
        shape_color=np.asarray(color.numpy(), np.float32) if color is not None else np.full((n, 3), 0.6, np.float32),
        mesh_vertices=np.concatenate(verts) if verts else np.zeros((0, 3), np.float32),
        mesh_faces=np.concatenate(faces) if faces else np.zeros((0, 3), np.int32),
        mesh_vstart=np.asarray(vstart, np.int64),
        mesh_fstart=np.asarray(fstart, np.int64),
    )


def logged(routine, name: str):
    def run(sim, session):
        model = sim.model
        shape_body = np.asarray(model.shape_body.numpy())
        body_names = _labels(getattr(model, "body_label", None))
        shape_names = _labels(getattr(model, "shape_label", None))
        rows: dict[str, list[np.ndarray]] = {}
        contacts = []

        def grab():
            c = sim.read()
            snap = sim.pipeline.snapshot
            row = {
                "step": np.int64(sim.step_count),
                "t": np.float64(sim.time),
                "body_q": np.asarray(c.body_q, np.float32),  # [B, 7] xyz + quat xyzw
                "body_qd": np.asarray(c.body_qd, np.float32),  # [B, 6] linear + angular
                "joint_q": np.asarray(snap.joint_q, np.float32),
                "joint_qd": np.asarray(snap.joint_qd, np.float32),
                "joint_target_q": np.asarray(c.joint_target_q, np.float32),
            }
            for k, v in row.items():
                rows.setdefault(k, []).append(np.copy(v))
            ct = c.contacts
            if ct is None:
                return
            if hasattr(ct, "update"):
                ct.update(sim.step_count)
            a, b = ct._shape_pairs()
            n = len(a)
            # Where each contact is and which way it pushes: MuJoCo's own contact rows, which the
            # probe read its geoms, forces and depths from, in the same order.
            pos = normal = None
            data = getattr(getattr(sim.contacts, "solver", None), "mjw_data", None)
            if data is not None and n:
                mjc = data.contact
                if np.array_equal(np.asarray(mjc.geom.numpy()[:n]), np.asarray(ct.geoms)):
                    pos = np.asarray(mjc.pos.numpy()[:n], dtype=np.float64)
                    normal = np.asarray(mjc.frame.numpy()[:n], dtype=np.float64).reshape(n, 3, 3)[:, 0]
            pairs = []
            for i in range(n):
                sa, sb = int(a[i]), int(b[i])
                force = np.asarray(ct.impulses[i], dtype=np.float64) / ct.dt  # [N] on b, world frame
                row = {
                    "a": shape_names[sa] if sa < len(shape_names) else sa,
                    "b": shape_names[sb] if sb < len(shape_names) else sb,
                    "impulse": [round(float(v), 6) for v in ct.impulses[i]],
                    "force": [round(float(v), 4) for v in force],
                    "depth": round(float(ct.depths[i]), 6),
                }
                if pos is not None:
                    nrm = normal[i]
                    fn = float(force @ nrm)
                    row["pos"] = [round(float(v), 5) for v in pos[i]]
                    row["normal"] = [round(float(v), 5) for v in nrm]  # from a toward b
                    row["normal_force"] = round(fn, 4)  # [N], along normal
                    row["friction_force"] = round(float(np.linalg.norm(force - fn * nrm)), 4)  # [N], tangential
                pairs.append(row)
            contacts.append({"step": int(sim.step_count), "t": round(float(sim.time), 4), "pairs": pairs})

        RUN.mkdir(exist_ok=True)
        _write_shapes(model, RUN / f"{name}_shapes.npz")
        grab()
        try:
            for _ in routine(sim, session):
                grab()
                yield
        finally:
            RUN.mkdir(exist_ok=True)
            np.savez_compressed(
                RUN / f"{name}_phys.npz",
                **{k: np.stack(v) for k, v in rows.items()},
                body_names=np.array(body_names),
                shape_names=np.array(shape_names),
                shape_body=shape_body,
                joint_names=np.array(_labels(getattr(model, "joint_label", None))),
            )
            with open(RUN / f"{name}_contacts.jsonl", "w") as f:
                for line in contacts:
                    f.write(json.dumps(line) + "\n")
            print(f"[twin] wrote run/{name}_phys.npz ({len(rows['step'])} steps) and run/{name}_contacts.jsonl", flush=True)

    return run
