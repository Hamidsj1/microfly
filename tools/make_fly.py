#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fly.py  --  procedural housefly (Musca domestica) -> fly.glb  (glTF 2.0 binary)

Only Python 3 + numpy.  Run:   python make_fly.py
Coordinates: right-handed, Y up, the fly faces +Z (head +Z, abdomen -Z), origin = thorax centre.
Fly's LEFT side is +X (left = up x forward = Y x Z = +X).  Right-side parts are exact mirrors.
"""
import json
import math
import os
import struct
import sys

import numpy as np

# =============================================================================
# 1. Settings
# =============================================================================
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fly.glb")
MAX_TRIS = 80000
SEED = 20261117
RNG = np.random.default_rng(SEED)

WING_LEN = 0.90          # hinge -> tip, in body lengths (body ~1.0)
WING_SWEEP_DEG = 28.0    # rest pose: wings swept back
HINGE_L = np.array([0.095, 0.105, 0.03])   # wing hinge on the thorax (left side, +X)
BODY_BRISTLES_THORAX = 760
BODY_BRISTLES_ABDOMEN = 900   # thorax + abdomen >= 1500 (asserted)

REQUIRED_NODES = ["Head", "Thorax", "Abdomen", "Eye_L", "Eye_R", "Haltere_L", "Haltere_R",
                  "Proboscis", "Antenna_L", "Antenna_R", "Wing_L", "Wing_R",
                  "Leg_L0", "Leg_L1", "Leg_L2", "Leg_R0", "Leg_R1", "Leg_R2"]


# =============================================================================
# 2. Vector utilities
# =============================================================================
def norm(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def hash01(P, seed=1.0):
    """cheap deterministic hash noise in [0,1) from 3D (or 2D) coordinates"""
    P = np.asarray(P, dtype=np.float64)
    k = np.array([12.9898, 78.233, 37.719])[: P.shape[-1]]
    v = np.sin(P @ k * seed) * 43758.5453
    return v - np.floor(v)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def basis_from_dir(d):
    """right-handed (a, b, d) with a x b = d"""
    d = norm(d)
    h = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    a = norm(np.cross(h, d))
    b = np.cross(d, a)
    return a, b, d


def lerp_profile(xs, ys, x, smooth=2):
    """piecewise-linear profile followed by a light moving-average smoothing"""
    x = np.asarray(x, dtype=np.float64)
    y = np.interp(x, xs, ys)
    if smooth:
        k = np.array([1, 2, 3, 2, 1], dtype=np.float64)
        k /= k.sum()
        for _ in range(smooth):
            pad = np.pad(y, 2, mode="edge")
            y = np.convolve(pad, k, mode="valid")
    return y


# =============================================================================
# 3. Mesh data structure
# =============================================================================
class Mesh:
    def __init__(self, P, I, UV=None, C=None):
        self.P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        self.I = np.asarray(I, dtype=np.int64).reshape(-1, 3)
        n = len(self.P)
        self.UV = np.zeros((n, 2)) if UV is None else np.asarray(UV, dtype=np.float64).reshape(-1, 2)
        self.C = np.ones((n, 4)) if C is None else np.asarray(C, dtype=np.float64).reshape(-1, 4)
        self.N = None

    def finish(self):
        self.N = compute_smooth_normals(self.P, self.I)
        return self

    def set_color(self, rgba):
        self.C = np.tile(np.asarray(rgba, dtype=np.float64), (len(self.P), 1))
        return self

    def translated(self, t):
        m = Mesh(self.P + np.asarray(t), self.I.copy(), self.UV.copy(), self.C.copy())
        m.N = None if self.N is None else self.N.copy()
        return m

    def transformed(self, R, t=(0, 0, 0)):
        m = Mesh(self.P @ np.asarray(R).T + np.asarray(t), self.I.copy(), self.UV.copy(), self.C.copy())
        m.N = None if self.N is None else norm(self.N @ np.asarray(R).T)
        return m

    def mirrored_x(self):
        """exact left/right mirror: x -> -x, triangle winding flipped"""
        P = self.P * np.array([-1.0, 1.0, 1.0])
        m = Mesh(P, self.I[:, [0, 2, 1]].copy(), self.UV.copy(), self.C.copy())
        if self.N is not None:
            m.N = self.N * np.array([-1.0, 1.0, 1.0])
        return m


def merge_meshes(meshes):
    meshes = [m for m in meshes if m is not None and len(m.I)]
    P, I, UV, C, N = [], [], [], [], []
    off = 0
    have_n = all(m.N is not None for m in meshes)
    for m in meshes:
        P.append(m.P)
        I.append(m.I + off)
        UV.append(m.UV)
        C.append(m.C)
        if have_n:
            N.append(m.N)
        off += len(m.P)
    out = Mesh(np.vstack(P), np.vstack(I), np.vstack(UV), np.vstack(C))
    if have_n:
        out.N = np.vstack(N)
    return out


# =============================================================================
# 4. Geometry helpers
# =============================================================================
def strip_faces(nr, k):
    """faces between consecutive rings; rings have k+1 columns (seam duplicated).
    Ring i runs CCW (theta increasing) about the direction of travel d, so
    (v00, v01, v10) has normal  theta_hat x d_hat = outward."""
    i = np.arange(nr - 1)[:, None]
    j = np.arange(k)[None, :]
    v00 = (i * (k + 1) + j).ravel()
    v01 = v00 + 1
    v10 = v00 + (k + 1)
    v11 = v10 + 1
    return np.concatenate([np.stack([v00, v01, v10], 1), np.stack([v01, v11, v10], 1)])


def loft(rings, colors=None, uv_v=None, caps=True):
    """rings: (nr, k+1, 3) ring points (last column == first column).  Travel direction right-handed
    with the ring orientation (see strip_faces).  Optionally closes both ends with triangle fans."""
    rings = np.asarray(rings, dtype=np.float64)
    nr, k1, _ = rings.shape
    k = k1 - 1
    P = rings.reshape(-1, 3)
    u = np.linspace(0, 1, k1)[None, :].repeat(nr, 0)
    v = (np.linspace(0, 1, nr) if uv_v is None else np.asarray(uv_v))[:, None].repeat(k1, 1)
    UV = np.stack([u, v], -1).reshape(-1, 2)
    C = np.ones((len(P), 4)) if colors is None else np.asarray(colors, dtype=np.float64).reshape(-1, 4)
    parts = [Mesh(P, strip_faces(nr, k), UV, C)]
    if caps:
        for end, flip in ((0, True), (nr - 1, False)):
            ring = rings[end, :k]
            c = ring.mean(0)
            Pc = np.vstack([c[None], ring, ring[:1]])
            f = np.array([[0, j + 1, j + 2] for j in range(k)])
            if flip:
                f = f[:, [0, 2, 1]]
            Cc = np.tile(C[end * k1], (len(Pc), 1))
            parts.append(Mesh(Pc, f, np.zeros((len(Pc), 2)), Cc))
    return merge_meshes(parts)


def compute_smooth_normals(P, I, weld=True):
    """area weighted vertex normals; positionally coincident vertices (seams) are welded so
    shading is continuous across UV seams."""
    P = np.asarray(P, dtype=np.float64)
    tri = P[I]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if weld:
        key = np.round(P * 1e5).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        acc = np.zeros((inv.max() + 1, 3))
        for c in range(3):
            np.add.at(acc, inv[I[:, c]], fn)
        N = acc[inv]
    else:
        N = np.zeros_like(P)
        for c in range(3):
            np.add.at(N, I[:, c], fn)
    ln = np.linalg.norm(N, axis=1, keepdims=True)
    N = np.where(ln > 1e-14, N / np.maximum(ln, 1e-14), np.array([0.0, 1.0, 0.0]))
    return N


def ellipsoid(center, radii, axis="z", nu=28, nv=18, color=(1, 1, 1, 1)):
    """ellipsoid whose poles lie on `axis`.  radii are given as (rx, ry, rz)."""
    rx, ry, rz = radii
    if axis == "z":
        A, B, D, ra, rb, rd = (1, 0, 0), (0, 1, 0), (0, 0, 1), rx, ry, rz
    elif axis == "y":
        A, B, D, ra, rb, rd = (0, 0, 1), (1, 0, 0), (0, 1, 0), rz, rx, ry
    else:
        A, B, D, ra, rb, rd = (0, 1, 0), (0, 0, 1), (1, 0, 0), ry, rz, rx
    A, B, D = map(np.array, (A, B, D))
    th = np.linspace(0, 2 * np.pi, nu + 1)
    phi = np.linspace(-np.pi / 2, np.pi / 2, nv + 1)
    cph, sph = np.cos(phi)[:, None, None], np.sin(phi)[:, None, None]
    ring = (cph * (ra * np.cos(th)[None, :, None] * A + rb * np.sin(th)[None, :, None] * B)
            + sph * rd * D)
    pts = np.asarray(center) + ring
    colors = np.tile(np.asarray(color, dtype=np.float64), (pts.shape[0] * pts.shape[1], 1))
    return loft(pts, colors, caps=False)


def tube(path, radii, sides=8, color=(1, 1, 1, 1), caps=True):
    """tube along a polyline with parallel-transport frames; radii per path point."""
    path = np.asarray(path, dtype=np.float64)
    n = len(path)
    radii = np.broadcast_to(np.asarray(radii, dtype=np.float64), (n,))
    tang = np.zeros_like(path)
    tang[1:-1] = path[2:] - path[:-2]
    tang[0] = path[1] - path[0]
    tang[-1] = path[-1] - path[-2]
    tang = norm(tang)
    A = np.zeros((n, 3))
    B = np.zeros((n, 3))
    a0, b0, _ = basis_from_dir(tang[0])
    A[0], B[0] = a0, b0
    for i in range(1, n):
        a = A[i - 1] - tang[i] * np.dot(A[i - 1], tang[i])
        if np.linalg.norm(a) < 1e-9:
            a = basis_from_dir(tang[i])[0]
        A[i] = norm(a)
        B[i] = np.cross(tang[i], A[i])
    th = np.linspace(0, 2 * np.pi, sides + 1)
    ring = (path[:, None, :] + radii[:, None, None] *
            (np.cos(th)[None, :, None] * A[:, None, :] + np.sin(th)[None, :, None] * B[:, None, :]))
    colors = np.tile(np.asarray(color, dtype=np.float64), (n * (sides + 1), 1))
    return loft(ring, colors, caps=caps)


def cones(bases, tips, radii, base_color, tip_color, sides=4):
    """many tapered spikes in one call (used for bristles).  Fully vectorised."""
    bases = np.asarray(bases, dtype=np.float64)
    tips = np.asarray(tips, dtype=np.float64)
    n = len(bases)
    radii = np.broadcast_to(np.asarray(radii, dtype=np.float64), (n,))
    d = norm(tips - bases)
    h = np.where((np.abs(d[:, 1]) < 0.9)[:, None], np.array([[0, 1.0, 0]]), np.array([[1.0, 0, 0]]))
    a = norm(np.cross(h, d))
    b = np.cross(d, a)
    th = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = (bases[:, None, :] + radii[:, None, None] *
            (np.cos(th)[None, :, None] * a[:, None, :] + np.sin(th)[None, :, None] * b[:, None, :]))
    P = np.concatenate([ring, tips[:, None, :]], axis=1).reshape(-1, 3)   # (n, sides+1)
    j = np.arange(sides)
    f = np.stack([j, (j + 1) % sides, np.full(sides, sides)], 1)          # (sides,3)
    I = (f[None] + (np.arange(n) * (sides + 1))[:, None, None]).reshape(-1, 3)
    C = np.empty((n, sides + 1, 4))
    C[:, :sides] = np.asarray(base_color)
    C[:, sides] = np.asarray(tip_color)
    UV = np.zeros((len(P), 2))
    UV.reshape(n, sides + 1, 2)[:, sides, 1] = 1.0
    return Mesh(P, I, UV, C.reshape(-1, 4))


def place_bristles(mesh, count, weights, length, radius, rng, back=0.55, base_c=(0.03, 0.03, 0.03, 1),
                   tip_c=(0.22, 0.20, 0.16, 1)):
    """exactly `count` distinct surface vertices are picked (weighted, no rejection filter),
    so the final bristle count is deterministic."""
    w = np.asarray(weights, dtype=np.float64)
    idx = rng.choice(len(mesh.P), size=count, replace=False, p=w / w.sum())
    N = mesh.N[idx]
    dirs = norm(N + np.array([0.0, 0.0, -back]) + rng.normal(0, 0.18, (count, 3)))
    L = rng.uniform(length[0], length[1], count)
    base = mesh.P[idx] - N * 0.002
    tip = mesh.P[idx] + dirs * L[:, None]
    return cones(base, tip, radius, base_c, tip_c)


def bristles_on_path(path, radii, count, length, rng, forward=0.6, radius=0.0016):
    """bristles along a limb: stratified positions along the path, random radial direction"""
    path = np.asarray(path, dtype=np.float64)
    radii = np.broadcast_to(np.asarray(radii, dtype=np.float64), (len(path),))
    s = (np.arange(count) + rng.random(count)) / count
    t = s * (len(path) - 1)
    i0 = np.clip(np.floor(t).astype(int), 0, len(path) - 2)
    f = (t - i0)[:, None]
    pos = path[i0] * (1 - f) + path[i0 + 1] * f
    rad = radii[i0] * (1 - f[:, 0]) + radii[i0 + 1] * f[:, 0]
    tang = norm(path[i0 + 1] - path[i0])
    out = np.empty((count, 3))
    for q in range(count):
        a, b, _ = basis_from_dir(tang[q])
        ang = rng.uniform(0, 2 * np.pi)
        out[q] = np.cos(ang) * a + np.sin(ang) * b
    dirs = norm(out + forward * tang)
    L = rng.uniform(length[0], length[1], count)
    return cones(pos + out * rad[:, None] * 0.7, pos + out * rad[:, None] * 0.7 + dirs * L[:, None], radius,
                 (0.03, 0.025, 0.02, 1), (0.25, 0.2, 0.15, 1), sides=3)


def orientation_score(mesh):
    """fraction of triangles whose normal points away from the mesh centroid (closed convex-ish bodies)"""
    tri = mesh.P[mesh.I]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ctr = tri.mean(1)
    ok = (fn * (ctr - mesh.P.mean(0))).sum(1)
    valid = np.linalg.norm(fn, axis=1) > 1e-14
    return float((ok[valid] > 0).mean())


# =============================================================================
# 5. Procedural colour helpers (colours are written to COLOR_0; sRGB-looking values, converted
#    to linear at write time)
# =============================================================================
def C4(r, g, b, a=1.0):
    return np.array([r, g, b, a], dtype=np.float64)


def mix(c0, c1, t):
    t = np.asarray(t, dtype=np.float64)[..., None]
    return np.asarray(c0) * (1 - t) + np.asarray(c1) * t


# =============================================================================
# 6. Materials (explicit per-part mapping; vertex colour carries the detail, so baseColorFactor
#    is white and nothing gets multiplied twice)
# =============================================================================
MATERIALS = {
    "cuticle": dict(metallic=0.0, rough=0.42),
    "abdomen": dict(metallic=0.0, rough=0.30),
    "eye":     dict(metallic=0.0, rough=0.22),
    "face":    dict(metallic=0.0, rough=0.45),
    "bristle": dict(metallic=0.0, rough=0.70),
    "wing":    dict(metallic=0.0, rough=0.06, blend=True),
    "vein":    dict(metallic=0.0, rough=0.50),
}


class Part:
    """one glTF node: translation = pivot, prims = [(mesh_in_pivot_local_coords, material_name)]"""

    def __init__(self, name, pivot=(0, 0, 0)):
        self.name = name
        self.pivot = np.asarray(pivot, dtype=np.float64)
        self.prims = []

    def add(self, mesh, mat):
        self.prims.append((mesh.translated(-self.pivot), mat))
        return self

    def mirrored(self, new_name):
        p = Part(new_name, self.pivot * np.array([-1.0, 1.0, 1.0]))
        p.prims = [(m.mirrored_x(), mat) for m, mat in self.prims]
        return p


# =============================================================================
# 7. Thorax
# =============================================================================
def build_thorax():
    zs = np.linspace(-0.215, 0.235, 64)
    kz = [-0.215, -0.17, -0.09, 0.0, 0.09, 0.17, 0.215, 0.235]
    rx = lerp_profile(kz, [0.05, 0.115, 0.15, 0.162, 0.150, 0.118, 0.07, 0.035], zs)
    ry = lerp_profile(kz, [0.05, 0.11, 0.150, 0.165, 0.150, 0.115, 0.07, 0.035], zs)
    cy = lerp_profile(kz, [0.00, 0.012, 0.03, 0.038, 0.030, 0.015, 0.0, -0.01], zs)
    k = 36
    th = np.linspace(0, 2 * np.pi, k + 1)
    ca, sa = np.cos(th)[None, :], np.sin(th)[None, :]
    # flatter underside than back
    yy = ry[:, None] * sa * np.where(sa < 0, 0.72, 1.0)
    rings = np.stack([rx[:, None] * ca * np.ones_like(sa), cy[:, None] + yy, zs[:, None] * np.ones_like(ca)], -1)
    m = loft(rings).finish()

    # scutellum: rounded bump at the rear-dorsal end
    scut = ellipsoid((0, 0.083, -0.165), (0.075, 0.042, 0.052), "z", 22, 12)
    scut = scut.transformed(np.array([[1, 0, 0], [0, math.cos(-0.35), -math.sin(-0.35)],
                                      [0, math.sin(-0.35), math.cos(-0.35)]]), (0, 0, 0))
    scut.finish()
    body = merge_meshes([m, scut]).finish()
    P, N = body.P, body.N

    # colour: grey-black cuticle, 4 dark longitudinal stripes on the back, lighter dusted flanks
    ax = np.abs(P[:, 0]) / 0.16
    dorsal = smoothstep(0.15, 0.55, N[:, 1])
    in_front = smoothstep(-0.13, -0.08, P[:, 2]) * (1 - smoothstep(0.17, 0.215, P[:, 2]))
    stripe = ((ax > 0.07) & (ax < 0.33)).astype(float) + ((ax > 0.52) & (ax < 0.72)).astype(float)
    stripe = np.clip(stripe, 0, 1) * dorsal * in_front
    base = mix(C4(0.40, 0.41, 0.42), C4(0.54, 0.55, 0.55), smoothstep(0.1, 0.9, np.abs(N[:, 0])))
    under = smoothstep(0.0, -0.6, N[:, 1])
    base = mix(base, C4(0.30, 0.30, 0.30), under)
    base = mix(base, C4(0.05, 0.05, 0.055), stripe)
    base *= (0.90 + 0.20 * hash01(np.round(P * 90), 3.7))[:, None] * np.array([1, 1, 1, 1.0])
    body.C = np.clip(np.hstack([base[:, :3], np.ones((len(P), 1))]), 0, 1)
    scut_mask = (P[:, 2] < -0.12) & (P[:, 1] > 0.06)
    body.C[scut_mask, :3] = body.C[scut_mask, :3] * 0.55 + np.array([0.12, 0.12, 0.12]) * 0.45

    w = 0.25 + 1.75 * np.clip(N[:, 1], 0, 1) + 0.25 * stripe
    bristles = place_bristles(body, BODY_BRISTLES_THORAX, w, (0.020, 0.046), 0.0020, RNG)
    bristles.finish()
    return body, bristles


# =============================================================================
# 8. Abdomen  (5 visible segments, overlapping tergites, tan flanks + dark midline stripe)
# =============================================================================
ABD_Z0, ABD_Z1 = -0.545, -0.13      # rear tip -> front (inside the thorax)
ABD_SEGS = 5
ABD_BODY_Z0 = -0.51                 # end of the 5th segment (rear), tip cap beyond it
ABD_SEG_LEN = (ABD_Z1 - ABD_BODY_Z0) / ABD_SEGS


def abd_seg_t(z):
    """segment index and position t in [0,1) inside a segment for a given z (rear -> front)"""
    q = (np.asarray(z) - ABD_BODY_Z0) / ABD_SEG_LEN
    return np.floor(np.clip(q, 0, ABD_SEGS - 1e-6)), np.clip(q - np.floor(q), 0, 1)


def build_abdomen():
    # ring z list: tip cap rings + for each segment a sawtooth sequence (step at segment borders)
    z_list = [ABD_Z0, ABD_Z0 + 0.012, ABD_Z0 + 0.026]
    f_list = [0.10, 0.45, 0.80]
    per = 7
    for s in range(ABD_SEGS):
        zs0 = ABD_BODY_Z0 + s * ABD_SEG_LEN
        for q in range(per):
            t = q / (per - 1)
            z_list.append(zs0 + 0.004 + t * (ABD_SEG_LEN - 0.006))
            f_list.append(0.935 + 0.065 * t ** 1.4)          # tergite swells towards its front edge
    z = np.array(z_list)
    f = np.array(f_list)
    kz = [-0.545, -0.50, -0.42, -0.32, -0.22, -0.13]
    base_r = lerp_profile(kz, [0.012, 0.075, 0.105, 0.118, 0.108, 0.095], z, smooth=1)
    rx = base_r * f
    ry = base_r * f * 0.88
    cy = np.interp(z, [-0.545, -0.3, -0.13], [-0.012, -0.012, 0.0])
    k = 36
    th = np.linspace(0, 2 * np.pi, k + 1)
    ca, sa = np.cos(th)[None, :], np.sin(th)[None, :]
    rings = np.stack([rx[:, None] * ca, cy[:, None] + ry[:, None] * sa * np.where(sa < 0, 0.85, 1.0),
                      z[:, None] * np.ones_like(ca)], -1)
    m = loft(rings).finish()
    P, N = m.P, m.N
    seg, t = abd_seg_t(P[:, 2])

    tan = C4(0.80, 0.63, 0.30)
    cream = C4(0.86, 0.80, 0.62)
    dark = C4(0.10, 0.075, 0.05)
    col = mix(tan, cream, smoothstep(0.0, -0.55, N[:, 1]))                 # paler belly
    col = mix(col, C4(0.50, 0.40, 0.22), smoothstep(0.25, 0.85, N[:, 1]) * 0.55)   # browner back
    mid = (1 - smoothstep(0.012, 0.030, np.abs(P[:, 0]))) * smoothstep(0.1, 0.5, N[:, 1])
    col = mix(col, dark, mid)
    band = smoothstep(0.70, 0.97, t) * smoothstep(-0.2, 0.5, N[:, 1])      # dark rear/front band per tergite
    col = mix(col, C4(0.16, 0.12, 0.07), band * 0.85)
    tip = smoothstep(-0.49, -0.545, P[:, 2])
    col = mix(col, C4(0.07, 0.06, 0.05), tip)
    col *= (0.93 + 0.14 * hash01(np.round(P * 110), 5.1))[:, None]
    m.C = np.hstack([np.clip(col[:, :3], 0, 1), np.ones((len(P), 1))])

    edge = smoothstep(0.72, 0.98, t) + 0.3
    w = (0.25 + 1.0 * np.clip(N[:, 1], 0, 1) + 0.6 * np.abs(N[:, 0])) * (0.4 + 5.0 * edge) * (P[:, 2] < -0.14)
    w = np.maximum(w, 1e-6)
    bristles = place_bristles(m, BODY_BRISTLES_ABDOMEN, w, (0.016, 0.038), 0.0018, RNG, back=0.7)
    bristles.finish()
    return m, bristles


# =============================================================================
# 9. Head and compound eyes
# =============================================================================
HEAD_C = np.array([0.0, 0.012, 0.318])
EYE_C = np.array([0.071, 0.030, 0.322])
EYE_R = (0.056, 0.086, 0.072)
FACET = 0.19     # facet size in radians (lattice unit)


def build_head():
    m = ellipsoid(HEAD_C, (0.082, 0.092, 0.074), "z", 36, 24).finish()
    P, N = m.P, m.N
    rel = P - HEAD_C
    front = smoothstep(0.10, 0.55, rel[:, 2] / 0.074)           # face towards +Z
    lowfront = front * smoothstep(0.35, -0.2, rel[:, 1] / 0.092)
    col = mix(C4(0.15, 0.15, 0.15), C4(0.88, 0.82, 0.64), front)           # pale face
    col = mix(col, C4(0.92, 0.80, 0.45), lowfront * 0.6)                  # yellowish mouth/jowl area
    top = smoothstep(0.55, 0.9, rel[:, 1] / 0.092)
    col = mix(col, C4(0.12, 0.12, 0.13), top * (1 - front * 0.6))          # dark vertex
    mid = (1 - smoothstep(0.004, 0.016, np.abs(P[:, 0]))) * front * smoothstep(0.0, 0.6, rel[:, 1] / 0.092)
    col = mix(col, C4(0.32, 0.22, 0.10), mid * 0.7)                        # frontal stripe
    m.C = np.hstack([col[:, :3], np.ones((len(P), 1))])
    return m


def hex_cell(q):
    """q (n,2) in lattice units (hex centre spacing 1).  returns distance to nearest hex centre and
    |dA-dB| which is ~0 on cell borders."""
    s3 = math.sqrt(3.0)
    x, y = q[:, 0], q[:, 1]
    dA = np.hypot(x - np.round(x), y - np.round(y / s3) * s3)
    xb, yb = x - 0.5, y - 0.5 * s3
    dB = np.hypot(xb - np.round(xb), yb - np.round(yb / s3) * s3)
    return np.minimum(dA, dB), np.abs(dA - dB)


def build_eye_left():
    """left eye (+X).  Ommatidia: real hexagonal dome bumps displaced along the normal + darker borders."""
    m = ellipsoid(EYE_C, EYE_R, "y", 84, 60).finish()
    loc = (m.P - EYE_C) / np.array(EYE_R)
    az = np.arctan2(loc[:, 2], loc[:, 0])
    el = np.arcsin(np.clip(loc[:, 1], -1, 1))
    q = np.stack([az * np.cos(el) / FACET, el / FACET], 1)
    d, edge = hex_cell(q)
    bump = 1.0 - smoothstep(0.0, 0.62, d)
    m.P = m.P + m.N * (0.0034 * bump)[:, None]
    m.finish()
    border = 1.0 - smoothstep(0.0, 0.16, edge)
    cell_id = np.floor(q * 1.0)
    tint = 0.88 + 0.24 * hash01(np.hstack([cell_id, np.zeros((len(q), 1))]), 2.3)
    base = mix(C4(0.40, 0.045, 0.03), C4(0.66, 0.16, 0.06), smoothstep(-0.2, 0.9, loc[:, 1]) * 0.5)
    base = mix(base, C4(0.62, 0.20, 0.10), smoothstep(0.5, 1.0, np.cos(az) * np.cos(el)) * 0.25)
    col = base * tint[:, None] * (1.0 - 0.55 * border)[:, None] * (0.7 + 0.5 * bump)[:, None]
    m.C = np.hstack([np.clip(col[:, :3], 0, 1), np.ones((len(m.P), 1))])
    return m


# =============================================================================
# 10. Proboscis and antennae
# =============================================================================
def build_proboscis():
    """rostrum + haustellum bending down/forward, ending in two flattened labella"""
    path = np.array([[0.0, -0.040, 0.372], [0.0, -0.072, 0.392], [0.0, -0.105, 0.410], [0.0, -0.128, 0.430],
                     [0.0, -0.140, 0.452]])
    r = np.array([0.020, 0.016, 0.012, 0.0105, 0.009])
    col = (0.62, 0.52, 0.34, 1.0)
    stalk = tube(path, r, 10, col)
    lab = [ellipsoid((0.0105 * s, -0.146, 0.467), (0.0115, 0.0045, 0.020), "z", 14, 8,
                     (0.90, 0.84, 0.66, 1.0)) for s in (1, -1)]
    return merge_meshes([stalk] + lab).finish()


def build_antenna_left():
    """scape/pedicel + flattened funiculus (3rd segment) + feathery arista"""
    base = np.array([0.020, 0.072, 0.382])
    ped = tube([base, base + [0.0035, -0.012, 0.016]], [0.0065, 0.0060], 8, (0.30, 0.22, 0.12, 1))
    fun_c = base + np.array([0.007, -0.034, 0.028])
    fun = ellipsoid(fun_c, (0.0075, 0.030, 0.0145), "y", 16, 10, (0.24, 0.17, 0.10, 1))
    a0 = base + np.array([0.007, -0.016, 0.040])
    arista_path = np.array([a0 + [0.0, 0.0, 0.0], a0 + [0.004, -0.014, 0.020], a0 + [0.011, -0.034, 0.036],
                            a0 + [0.020, -0.056, 0.046]])
    arista = tube(arista_path, [0.0018, 0.0014, 0.0010, 0.0006], 5, (0.06, 0.05, 0.04, 1))
    # feathery side hairs (both sides of the arista, in the antenna's plane)
    hairs = []
    nh = 9
    for sgn in (1.0, -1.0):
        t = (np.arange(nh) + 0.5) / nh
        pos = np.array([np.interp(t * 3, [0, 1, 2, 3], arista_path[:, c]) for c in range(3)]).T
        tang = norm(np.gradient(arista_path, axis=0).mean(0))
        side = norm(np.cross(tang, [1.0, 0.0, 0.0])) * sgn
        tips = pos + side * 0.013 * (1 - 0.5 * t)[:, None] + tang * 0.004
        hairs.append(cones(pos, tips, 0.0006, (0.05, 0.05, 0.05, 1), (0.30, 0.28, 0.22, 1), sides=3))
    return merge_meshes([ped, fun, arista] + hairs).finish()


# =============================================================================
# 11. Wing membrane and venation  (surface grid + vein tubes, built for the LEFT side, mirrored for right)
# =============================================================================
def wing_chord(s):
    s = np.asarray(s, dtype=np.float64)
    return 0.075 * (1 - s) + 0.30 * np.sin(np.pi * np.clip(s, 0, 1) ** 0.82) ** 0.62


def wing_point(s, t, side=1.0):
    """point on the wing surface in HINGE-LOCAL coordinates.  s: 0 base .. 1 tip, t: 0 leading .. 1 trailing edge.
    x = side * span * s   (mirroring done through `side`)."""
    s = np.asarray(s, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    c = wing_chord(s)
    span = WING_LEN
    z_le = 0.5 * wing_chord(0.0) - 0.020 * s
    z = z_le - t * c
    y = -0.030 * s ** 2 + 0.012 * np.sin(np.pi * t) * s
    x = side * span * s
    pts = np.stack([x, y, z], -1)
    # rest pose: swept back about Y (mirrored with side so left/right stay exact mirror images)
    R = rot_y(side * math.radians(WING_SWEEP_DEG))
    return pts @ R.T


def build_wing_left():
    nu, nv = 46, 16
    s = np.linspace(0, 1, nu + 1)
    t = np.linspace(0, 1, nv + 1)
    S, T = np.meshgrid(s, t, indexing="ij")
    P = wing_point(S, T, 1.0).reshape(-1, 3)
    I = []
    for i in range(nu):
        for j in range(nv):
            a = i * (nv + 1) + j
            b = a + (nv + 1)
            I.append([a, b, a + 1])
            I.append([a + 1, b, b + 1])
    UV = np.stack([S.ravel(), T.ravel()], 1)
    # colour: yellowish base -> clear tip, mild iridescence (thin film look through vertex colour),
    # alpha lives in COLOR_0 (material baseColorFactor is white)
    sf, tf = S.ravel(), T.ravel()
    ph = 2 * np.pi * (0.6 * sf + 1.4 * tf)
    irid = 0.07 * np.stack([np.sin(ph), np.sin(ph + 2.1), np.sin(ph + 4.2)], 1)
    base = mix(C4(0.96, 0.86, 0.50), C4(0.80, 0.87, 0.93), smoothstep(0.0, 0.55, sf))[:, :3]
    rgb = np.clip(base + irid * smoothstep(0.1, 0.5, sf)[:, None], 0, 1)
    alpha = 0.16 + 0.34 * (1 - sf) ** 2 + 0.10 * (1 - smoothstep(0.0, 0.08, tf)) + 0.05 * smoothstep(0.92, 1.0, tf)
    C = np.hstack([rgb, np.clip(alpha, 0, 1)[:, None]])
    mem = Mesh(P, I, UV, C).finish()

    # --- veins (Musca domestica-like layout: costa, subcosta, R1, R2+3, R4+5, M, CuA, anal + crossveins)
    def vein(s0, s1, tfun, radius, n=36, lift=0.0030):
        ss = np.linspace(s0, s1, n)
        tt = np.broadcast_to(np.asarray(tfun(ss), dtype=np.float64), ss.shape)
        pts = wing_point(ss, tt, 1.0)
        pts[:, 1] += lift
        rr = np.linspace(radius, radius * 0.55, n)
        return tube(pts, rr, 5, (0.30, 0.20, 0.10, 1.0))

    v = []
    v.append(vein(0.0, 1.0, lambda s: 0.0 * s + 0.004, 0.0062, 48))                       # costa (leading edge)
    v.append(vein(0.04, 0.46, lambda s: 0.050 + 0.04 * s, 0.0040))                         # subcosta
    v.append(vein(0.05, 0.84, lambda s: 0.105 - 0.05 * s ** 2, 0.0042))                    # R1
    v.append(vein(0.05, 0.93, lambda s: 0.185 + 0.03 * s - 0.09 * s ** 3, 0.0040))        # R2+3
    v.append(vein(0.05, 0.985, lambda s: 0.275 + 0.025 * s - 0.20 * s ** 4, 0.0042))       # R4+5 (bends to the apex)
    # M: straight, then the characteristic forward bend near the margin (M1+2)
    v.append(vein(0.07, 1.0, lambda s: 0.47 + 0.04 * s - 0.30 * np.clip(s - 0.62, 0, 1) ** 1.2 * 1.6, 0.0040))
    v.append(vein(0.05, 0.98, lambda s: 0.62 + 0.06 * s + 0.08 * s ** 3, 0.0036))          # CuA
    v.append(vein(0.05, 0.62, lambda s: 0.82 + 0.05 * s, 0.0032))                          # anal vein
    # crossveins: r-m (between R4+5 and M) and dm-cu (between M and CuA)
    def cross(sc, t0, t1, radius=0.0030, n=8):
        tt = np.linspace(t0, t1, n)
        pts = wing_point(np.full(n, sc), tt, 1.0)
        pts[:, 1] += 0.0030
        return tube(pts, radius, 5, (0.30, 0.20, 0.10, 1.0))
    sc1 = 0.44
    t_r45 = 0.275 + 0.025 * sc1 - 0.20 * sc1 ** 4
    t_m = 0.47 + 0.04 * sc1
    v.append(cross(sc1, t_r45, t_m))
    sc2 = 0.66
    t_m2 = 0.47 + 0.04 * sc2 - 0.30 * max(sc2 - 0.62, 0) ** 1.2 * 1.6
    t_cu = 0.62 + 0.06 * sc2 + 0.08 * sc2 ** 3
    v.append(cross(sc2, t_m2, t_cu))
    veins = merge_meshes(v).finish()
    return mem, veins


# =============================================================================
# 12. Halteres
# =============================================================================
def build_haltere_left():
    stalk = tube([[0, 0, 0], [0.020, -0.006, -0.030], [0.036, -0.014, -0.052]], [0.0042, 0.0038, 0.0045], 8,
                 (0.92, 0.88, 0.72, 1))
    knob = ellipsoid((0.040, -0.016, -0.060), (0.0125, 0.0125, 0.017), "z", 16, 10, (0.95, 0.92, 0.80, 1))
    return merge_meshes([stalk, knob]).finish()


# =============================================================================
# 13. Six legs: coxa, trochanter, femur, tibia, 5-segment tarsus, claw pair, pulvilli
# =============================================================================
LEG_SPECS = [   # (hip on thorax, ankle target, femur len, tibia len, tarsus direction)
    dict(hip=(0.082, -0.092, 0.150), ankle=(0.200, -0.215, 0.405), lf=0.150, lt=0.165, tdir=(0.15, -0.70, 0.70)),
    dict(hip=(0.098, -0.100, 0.020), ankle=(0.340, -0.215, 0.080), lf=0.170, lt=0.190, tdir=(0.55, -0.80, 0.0)),
    dict(hip=(0.088, -0.100, -0.125), ankle=(0.270, -0.215, -0.265), lf=0.175, lt=0.200, tdir=(0.30, -0.70, -0.65)),
]
TARSUS_LENS = [0.040, 0.024, 0.019, 0.015, 0.020]


def two_bone_ik(root, target, l1, l2, pole):
    d_vec = target - root
    d = np.linalg.norm(d_vec)
    d = min(d, l1 + l2 - 1e-4)
    dirn = norm(d_vec)
    a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = math.sqrt(max(l1 * l1 - a * a, 1e-9))
    perp = pole - dirn * np.dot(pole, dirn)
    perp = norm(perp)
    knee = root + dirn * a + perp * h
    ankle = root + dirn * d
    return knee, ankle


def build_leg_left(idx, spec):
    hip = np.array(spec["hip"])
    out = norm([0.8, -0.35, 0.0])
    coxa_end = hip + out * 0.040 + np.array([0.0, 0.0, 0.0])
    troch_end = coxa_end + norm([0.5, -0.25, 0.1]) * 0.020
    knee, ankle = two_bone_ik(troch_end, np.array(spec["ankle"]), spec["lf"], spec["lt"],
                              norm([0.35, 1.0, 0.10]))
    ca = (0.20, 0.17, 0.12, 1)          # coxa
    cf = (0.17, 0.14, 0.09, 1)          # femur
    ct = (0.22, 0.17, 0.09, 1)          # tibia
    cta = (0.66, 0.55, 0.36, 1)         # tarsus (pale)
    parts = []
    parts.append(tube([hip, coxa_end], [0.0155, 0.0135], 10, ca))
    parts.append(ellipsoid(coxa_end, (0.0125, 0.0125, 0.0125), "z", 12, 8, ca))
    parts.append(tube([coxa_end, troch_end], [0.0120, 0.0110], 8, ca))
    fem_path = np.array([troch_end + (knee - troch_end) * u for u in np.linspace(0, 1, 5)])
    parts.append(tube(fem_path, [0.0120, 0.0135, 0.0130, 0.0105, 0.0080], 10, cf))
    parts.append(ellipsoid(knee, (0.0085, 0.0085, 0.0085), "z", 12, 8, cf))
    tib_path = np.array([knee + (ankle - knee) * u for u in np.linspace(0, 1, 6)])
    tib_r = [0.0075, 0.0072, 0.0064, 0.0056, 0.0050, 0.0045]
    parts.append(tube(tib_path, tib_r, 8, ct))
    # tarsus: 5 sub-segments with small joint swellings, continuing along tdir
    td = norm(spec["tdir"])
    p = ankle.copy()
    for k, ln in enumerate(TARSUS_LENS):
        q = p + td * ln
        parts.append(tube([p, q], [0.0042 - 0.0004 * k, 0.0036 - 0.0004 * k], 7, cta))
        parts.append(ellipsoid(q, (0.0036 - 0.0003 * k,) * 3, "z", 8, 6, (0.58, 0.47, 0.30, 1)))
        p = q
    foot = p
    # claws (pair) and pulvilli (adhesive pads)
    a, b, _ = basis_from_dir(td)
    for sgn in (1, -1):
        c0 = foot
        c1 = foot + td * 0.006 + a * 0.0045 * sgn - np.array([0, 0.004, 0])
        c2 = c1 + td * 0.005 + a * 0.003 * sgn - np.array([0, 0.005, 0])
        parts.append(tube([c0, c1, c2], [0.0020, 0.0013, 0.0004], 5, (0.05, 0.04, 0.03, 1)))
        parts.append(ellipsoid(foot + td * 0.004 + a * 0.0055 * sgn - np.array([0, 0.003, 0]),
                               (0.0050, 0.0022, 0.0070), "z", 10, 6, (0.93, 0.88, 0.74, 1)))
    leg = merge_meshes(parts).finish()
    bris = [bristles_on_path(fem_path, [0.012, 0.0135, 0.013, 0.0105, 0.008], 16, (0.012, 0.028), RNG),
            bristles_on_path(tib_path, tib_r, 26, (0.010, 0.026), RNG, forward=0.8)]
    bristles = merge_meshes(bris).finish()
    return hip, leg, bristles


# =============================================================================
# 14. Scene graph / node hierarchy
# =============================================================================
def build_parts():
    parts = []
    thorax, th_bris = build_thorax()
    abdomen, ab_bris = build_abdomen()
    n_bristles = BODY_BRISTLES_THORAX + BODY_BRISTLES_ABDOMEN
    assert n_bristles >= 1500, "need >= 1500 thorax+abdomen bristles"

    p = Part("Thorax").add(thorax, "cuticle").add(th_bris, "bristle")
    parts.append(p)
    parts.append(Part("Abdomen").add(abdomen, "abdomen").add(ab_bris, "bristle"))
    parts.append(Part("Head").add(build_head(), "face"))

    eye_l = build_eye_left()
    parts.append(Part("Eye_L", EYE_C).add(eye_l, "eye"))
    parts.append(Part("Eye_R", EYE_C * np.array([-1, 1, 1])).add(eye_l.mirrored_x(), "eye"))

    prob = build_proboscis()
    parts.append(Part("Proboscis", (0.0, -0.040, 0.372)).add(prob, "face"))

    ant = build_antenna_left()
    pl = Part("Antenna_L", (0.020, 0.072, 0.382)).add(ant, "cuticle")
    parts.append(pl)
    parts.append(pl.mirrored("Antenna_R"))

    mem, veins = build_wing_left()
    wl = Part("Wing_L", HINGE_L)
    wl.prims = [(mem, "wing"), (veins, "vein")]        # already hinge-local (origin = hinge)
    parts.append(wl)
    parts.append(wl.mirrored("Wing_R"))

    hal = build_haltere_left()
    hl = Part("Haltere_L", (0.072, 0.040, -0.150))
    hl.prims = [(hal, "face")]
    parts.append(hl)
    parts.append(hl.mirrored("Haltere_R"))

    for i, spec in enumerate(LEG_SPECS):
        hip, leg, bris = build_leg_left(i, spec)
        pl = Part("Leg_L%d" % i, hip).add(leg, "cuticle").add(bris, "bristle")
        parts.append(pl)
        parts.append(pl.mirrored("Leg_R%d" % i))
    order = {n: i for i, n in enumerate(REQUIRED_NODES)}
    parts.sort(key=lambda q: order[q.name])
    return parts


# =============================================================================
# 15. GLB binary writer
# =============================================================================
def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def pad4(b, fill=b"\x00"):
    return b + fill * ((4 - len(b) % 4) % 4)


def write_glb(parts, path):
    mat_names = list(MATERIALS.keys())
    materials = []
    for name in mat_names:
        spec = MATERIALS[name]
        m = {"name": name,
             "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0],   # white: vertex colour is the colour
                                      "metallicFactor": spec["metallic"], "roughnessFactor": spec["rough"]}}
        if spec.get("blend"):
            m["alphaMode"] = "BLEND"
            m["doubleSided"] = True
        else:
            m["alphaMode"] = "OPAQUE"
        materials.append(m)

    binbuf = bytearray()
    buffer_views, accessors, meshes, nodes = [], [], [], []

    def add_view(data, target):
        off = len(binbuf)
        binbuf.extend(data)
        while len(binbuf) % 4:
            binbuf.append(0)
        buffer_views.append({"buffer": 0, "byteOffset": off, "byteLength": len(data), "target": target})
        return len(buffer_views) - 1

    def add_acc(view, ctype, count, atype, mn=None, mx=None):
        a = {"bufferView": view, "componentType": ctype, "count": int(count), "type": atype}
        if mn is not None:
            a["min"], a["max"] = [float(x) for x in mn], [float(x) for x in mx]
        accessors.append(a)
        return len(accessors) - 1

    for part in parts:
        prims = []
        for mesh, mat in part.prims:
            if mesh.N is None:
                mesh.finish()
            P = mesh.P.astype(np.float32)
            N = mesh.N.astype(np.float32)
            UV = mesh.UV.astype(np.float32)
            C = mesh.C.copy()
            C[:, :3] = srgb_to_linear(np.clip(C[:, :3], 0, 1))      # COLOR_0 is linear in glTF
            C = C.astype(np.float32)
            I = mesh.I.astype(np.uint32).reshape(-1)
            a_pos = add_acc(add_view(P.tobytes(), 34962), 5126, len(P), "VEC3", P.min(0), P.max(0))
            a_nor = add_acc(add_view(N.tobytes(), 34962), 5126, len(N), "VEC3")
            a_uv = add_acc(add_view(UV.tobytes(), 34962), 5126, len(UV), "VEC2")
            a_col = add_acc(add_view(C.tobytes(), 34962), 5126, len(C), "VEC4")
            a_idx = add_acc(add_view(I.tobytes(), 34963), 5125, len(I), "SCALAR")
            prims.append({"attributes": {"POSITION": a_pos, "NORMAL": a_nor, "TEXCOORD_0": a_uv, "COLOR_0": a_col},
                          "indices": a_idx, "material": mat_names.index(mat), "mode": 4})
        meshes.append({"name": part.name + "_mesh", "primitives": prims})

    root_children = []
    nodes.append({"name": "Fly", "children": []})        # index 0 = root
    for i, part in enumerate(parts):
        n = {"name": part.name, "mesh": i}
        if np.linalg.norm(part.pivot) > 0:
            n["translation"] = [float(x) for x in part.pivot]
        nodes.append(n)
        root_children.append(len(nodes) - 1)             # each node added exactly once, one parent
    nodes[0]["children"] = root_children

    gltf = {"asset": {"version": "2.0", "generator": "make_fly.py (procedural Musca domestica)"},
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": nodes, "meshes": meshes, "materials": materials,
            "accessors": accessors, "bufferViews": buffer_views, "buffers": [{"byteLength": len(binbuf)}]}
    js = pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    bn = pad4(bytes(binbuf))
    total = 12 + 8 + len(js) + 8 + len(bn)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), b"JSON") + js)
        f.write(struct.pack("<I4s", len(bn), b"BIN\x00") + bn)
    return gltf


# =============================================================================
# 16. GLB reader / validator
# =============================================================================
def read_glb(path):
    data = open(path, "rb").read()
    magic, version, total = struct.unpack("<4sII", data[:12])
    off = 12
    js_len, js_type = struct.unpack("<I4s", data[off:off + 8])
    js = json.loads(data[off + 8: off + 8 + js_len].decode("utf-8"))
    off += 8 + js_len
    bn_len, bn_type = struct.unpack("<I4s", data[off:off + 8])
    bn = data[off + 8: off + 8 + bn_len]
    return data, magic, version, total, js, js_type, bn, bn_type


def node_matrix(n):
    """4x4 local matrix from TRS or matrix (extensible: translation, rotation quaternion, scale)"""
    if "matrix" in n:
        return np.array(n["matrix"], dtype=np.float64).reshape(4, 4).T
    t = np.array(n.get("translation", [0, 0, 0]), dtype=np.float64)
    q = np.array(n.get("rotation", [0, 0, 0, 1]), dtype=np.float64)
    s = np.array(n.get("scale", [1, 1, 1]), dtype=np.float64)
    x, y, z, w = q / np.linalg.norm(q)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4)
    M[:3, :3] = R * s
    M[:3, 3] = t
    return M


def validate(path):
    data, magic, version, total, js, js_type, bn, bn_type = read_glb(path)
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))
        assert cond, "VALIDATION FAILED: " + name

    check("magic == glTF", magic == b"glTF")
    check("version == 2", version == 2)
    check("totalLength == file size", total == len(data) == os.path.getsize(path))
    check("JSON chunk parsed / chunk types", js_type == b"JSON" and bn_type == b"BIN\x00")
    check("chunks 4-byte aligned", (len(data) % 4 == 0) and (len(bn) % 4 == 0))
    nodes = js["nodes"]
    names = [n["name"] for n in nodes]
    check("root node 'Fly' exists once", names.count("Fly") == 1)
    for rn in REQUIRED_NODES:
        check("node %s exists exactly once" % rn, names.count(rn) == 1)
    assert "Wing_L" in names and "Wing_R" in names
    # parent map: every node has at most one parent, all required nodes are direct children of Fly
    parent = {}
    for pi, n in enumerate(nodes):
        for c in n.get("children", []):
            check("node %d has a single parent" % c, c not in parent)
            parent[c] = pi
    fly_i = names.index("Fly")
    for rn in REQUIRED_NODES:
        check("%s is a direct child of Fly" % rn, parent.get(names.index(rn)) == fly_i)
    for nm, hinge in (("Wing_L", HINGE_L), ("Wing_R", HINGE_L * np.array([-1, 1, 1]))):
        tr = np.array(nodes[names.index(nm)].get("translation", [0, 0, 0]))
        check("%s translation == hinge" % nm, np.allclose(tr, hinge, atol=1e-6))
    # accessors / bufferViews / indices
    check("buffer byteLength == BIN size", js["buffers"][0]["byteLength"] <= len(bn))
    comp_size = {5126: 4, 5125: 4, 5123: 2}
    ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
    for ai, a in enumerate(js["accessors"]):
        bv = js["bufferViews"][a["bufferView"]]
        need = a["count"] * ncomp[a["type"]] * comp_size[a["componentType"]]
        check("accessor %d fits its bufferView" % ai, need <= bv["byteLength"])
        check("bufferView %d inside buffer" % a["bufferView"], bv["byteOffset"] + bv["byteLength"] <= len(bn))
        check("bufferView %d 4-byte aligned" % a["bufferView"], bv["byteOffset"] % 4 == 0)

    def read_acc(ai):
        a = js["accessors"][ai]
        bv = js["bufferViews"][a["bufferView"]]
        dt = {5126: "<f4", 5125: "<u4", 5123: "<u2"}[a["componentType"]]
        arr = np.frombuffer(bn, dtype=dt, count=a["count"] * ncomp[a["type"]], offset=bv["byteOffset"])
        return arr.reshape(a["count"], ncomp[a["type"]]) if a["type"] != "SCALAR" else arr

    tris = 0
    mins, maxs = np.full(3, np.inf), np.full(3, -np.inf)
    nodes_world = {}
    for ni, n in enumerate(nodes):
        if "mesh" not in n:
            continue
        M = node_matrix(n)            # parents (Fly) have identity; compose upward for generality
        p = parent.get(ni)
        while p is not None:
            M = node_matrix(nodes[p]) @ M
            p = parent.get(p)
        for prim in js["meshes"][n["mesh"]]["primitives"]:
            check("primitive mode 4", prim.get("mode") == 4)
            for attr in ("POSITION", "NORMAL", "TEXCOORD_0", "COLOR_0"):
                check("%s has %s" % (n["name"], attr), attr in prim["attributes"])
            pos = read_acc(prim["attributes"]["POSITION"])
            nor = read_acc(prim["attributes"]["NORMAL"])
            idx = read_acc(prim["indices"])
            check("%s: finite positions" % n["name"], np.isfinite(pos).all() and np.isfinite(nor).all())
            check("%s: indices in range" % n["name"], idx.max() < len(pos))
            check("%s: index count multiple of 3" % n["name"], len(idx) % 3 == 0)
            check("%s: material index valid" % n["name"], 0 <= prim["material"] < len(js["materials"]))
            ln = np.linalg.norm(nor, axis=1)
            check("%s: unit normals" % n["name"], np.abs(ln - 1).max() < 1e-3)
            tris += len(idx) // 3
            w = (np.hstack([pos, np.ones((len(pos), 1))]) @ M.T)[:, :3]
            mins = np.minimum(mins, w.min(0))
            maxs = np.maximum(maxs, w.max(0))
    check("triangles < %d" % MAX_TRIS, tris < MAX_TRIS)
    # wing material
    wm = [m for m in js["materials"] if m["name"] == "wing"][0]
    check("wing alphaMode BLEND + doubleSided", wm["alphaMode"] == "BLEND" and wm["doubleSided"] is True)

    print("=" * 64)
    print("file        :", os.path.abspath(path))
    print("GLB version :", version)
    print("file size   : %d bytes (%.1f KB)" % (len(data), len(data) / 1024))
    print("nodes       :", len(nodes), "| meshes:", len(js["meshes"]), "| materials:", len(js["materials"]))
    print("triangles   :", tris, "(from primitive index counts)")
    print("node names  :", ", ".join(names))
    print("bbox min    : %.3f %.3f %.3f" % tuple(mins))
    print("bbox max    : %.3f %.3f %.3f" % tuple(maxs))
    print("dimensions  : %.3f x %.3f x %.3f  (X width, Y height, Z length)" % tuple(maxs - mins))
    print("validation  : %d checks passed" % sum(1 for _, ok in results if ok))
    print("=" * 64)
    return tris, mins, maxs


# =============================================================================
# 17. Main
# =============================================================================
def main():
    parts = build_parts()
    tris_pre = sum(len(m.I) for p in parts for m, _ in p.prims)
    assert tris_pre < MAX_TRIS, "triangle budget exceeded before writing: %d" % tris_pre
    write_glb(parts, OUT_PATH)
    tris, _, _ = validate(OUT_PATH)
    assert tris == tris_pre, "triangle count mismatch (%d vs %d)" % (tris, tris_pre)
    print("OK: triangle count identical before write and after re-read (%d)" % tris)


if __name__ == "__main__":
    main()
