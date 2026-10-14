from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from damage_3d import _read_focal_exif
from diag import dlog


def _ray_cast_2d_points(
    points_2d: np.ndarray,
    verts: np.ndarray, faces: np.ndarray,
    R: np.ndarray, t: np.ndarray, K: np.ndarray,
    *,
    batch: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    K_inv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    R64 = np.asarray(R, dtype=np.float64)
    t64 = np.asarray(t, dtype=np.float64).reshape(3)
    R_T = R64.T
    cam_pos = (R_T @ (-t64)).astype(np.float64)

    N = int(len(points_2d))
    if N == 0 or len(faces) == 0:
        return np.array([], dtype=int), np.zeros((0, 3), dtype=np.float64)

    xs = points_2d[:, 0].astype(np.float64)
    ys = points_2d[:, 1].astype(np.float64)
    dirs_cam = K_inv @ np.stack([xs, ys, np.ones(N, dtype=np.float64)])
    dirs = (R_T @ dirs_cam).T
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / (norms + 1e-10)

    v0 = verts[faces[:, 0]].astype(np.float64)
    e1 = verts[faces[:, 1]].astype(np.float64) - v0
    e2 = verts[faces[:, 2]].astype(np.float64) - v0

    hits_idx: List[int] = []
    hits_p3d: List[np.ndarray] = []

    for start in range(0, N, batch):
        d = dirs[start:start + batch]
        B = int(len(d))
        h = np.cross(d[:, None, :], e2[None, :, :])
        a = np.einsum('fi,bfi->bf', e1, h)
        ok = np.abs(a) > 1e-8
        inv = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
        s = cam_pos - v0
        u = inv * np.einsum('fi,bfi->bf', s, h)
        q = np.cross(s[None, :, :], e1[None, :, :])
        v_t = inv * np.einsum('bi,bfi->bf', d, q)
        t_hit = inv * np.einsum('fi,bfi->bf', e2, q)
        valid = ok & (u >= 0) & (v_t >= 0) & (u + v_t <= 1) & (t_hit > 1e-6)
        for bi in range(B):
            row = valid[bi]
            if not row.any():
                continue
            t_vals = np.where(row, t_hit[bi], np.inf)
            fi = int(t_vals.argmin())
            pt = cam_pos + t_vals[fi] * dirs[start + bi]
            hits_idx.append(start + bi)
            hits_p3d.append(pt)

    if not hits_idx:
        return np.array([], dtype=int), np.zeros((0, 3), dtype=np.float64)
    return (
        np.asarray(hits_idx, dtype=int),
        np.asarray(hits_p3d, dtype=np.float64),
    )


def _estimate_focal_near_px(
    near_path: Optional[str], Wn: int, K_far: np.ndarray, Wf: int,
) -> Tuple[float, str]:
    if near_path:
        try:
            f35mm, _, _, _ = _read_focal_exif(near_path)
            if f35mm and f35mm > 0:
                return float(f35mm) * float(Wn) / 36.0, 'exif_35mm'
        except Exception as exc:
            dlog(f'[near.focal] EXIF чтение упало: {exc!s}')
    focal_far_px = float(np.asarray(K_far, dtype=np.float64)[0, 0])
    return focal_far_px * (float(Wn) / float(Wf)), 'scaled_from_far'


def _render_damaged_faces_on_photo(
    img: np.ndarray,
    verts: np.ndarray, faces: np.ndarray,
    R: np.ndarray, t: np.ndarray, K: np.ndarray,
    damaged_face_idx,
    fill_color: Tuple[int, int, int] = (40, 40, 230),
    alpha: float = 0.55,
) -> np.ndarray:
    out = img.copy()
    if not damaged_face_idx:
        return out
    R64 = np.asarray(R, dtype=np.float64)
    t64 = np.asarray(t, dtype=np.float64).reshape(3)
    K64 = np.asarray(K, dtype=np.float64)

    cam = (R64 @ verts.T).T + t64
    front = cam[:, 2] > 1e-6
    if not front.any():
        return out
    uv = (K64 @ cam.T).T
    z = np.maximum(uv[:, 2:3], 1e-6)
    px = np.round(uv[:, :2] / z).astype(np.int32)

    h, w = out.shape[:2]
    sil = np.zeros((h, w), dtype=np.uint8)
    for fi in damaged_face_idx:
        face = faces[fi]
        if not (front[face[0]] and front[face[1]] and front[face[2]]):
            continue
        v0c = cam[face[0]]
        v1c = cam[face[1]]
        v2c = cam[face[2]]
        normal = np.cross(v1c - v0c, v2c - v0c)
        if normal[2] >= 0:
            continue
        tri = np.array([px[face[0]], px[face[1]], px[face[2]]], dtype=np.int32)
        cv2.fillConvexPoly(sil, tri, 1)

    mask = sil > 0
    if mask.any():
        col = np.array(fill_color, dtype=np.float32)
        out[mask] = (col * alpha + out[mask].astype(np.float32) * (1 - alpha)).astype(np.uint8)
    return out


def solve_near_pose_via_correspondences(
    pts_near: np.ndarray,
    pts_far: np.ndarray,
    R_far: np.ndarray, t_far: np.ndarray, K_far: np.ndarray,
    K_near: np.ndarray,
    mesh_verts: np.ndarray, mesh_faces: np.ndarray,
    *,
    ransac_reproj_err_px: float = 8.0,
    ransac_iterations: int = 500,
) -> Optional[Dict[str, Any]]:
    pts_near = np.asarray(pts_near, dtype=np.float64)
    pts_far = np.asarray(pts_far, dtype=np.float64)
    if len(pts_near) != len(pts_far) or len(pts_near) < 6:
        return None

    hit_idx, p3d = _ray_cast_2d_points(
        pts_far, mesh_verts, mesh_faces, R_far, t_far, K_far,
    )
    if len(hit_idx) < 6:
        dlog(
            f'[near.pnp] too few ray hits: {len(hit_idx)} '
            f'(out of {len(pts_far)} correspondences)'
        )
        return None

    obj_pts = p3d.reshape(-1, 1, 3).astype(np.float64)
    img_pts = pts_near[hit_idx].reshape(-1, 1, 2).astype(np.float64)
    K64 = np.asarray(K_near, dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, K64, dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=float(ransac_reproj_err_px),
            iterationsCount=int(ransac_iterations),
            confidence=0.999,
        )
    except cv2.error as exc:
        dlog(f'[near.pnp] solvePnPRansac error: {exc}')
        return None

    if not ok or inliers is None or len(inliers) < 4:
        dlog(
            f'[near.pnp] PnP failed: ok={ok} inliers='
            f'{0 if inliers is None else len(inliers)}'
        )
        return None

    inl_idx = inliers.flatten().astype(int).tolist()
    obj_inl = obj_pts[inl_idx]
    img_inl = img_pts[inl_idx]

    try:
        rvec_r, tvec_r = cv2.solvePnPRefineLM(
            obj_inl, img_inl, K64, dist, rvec.copy(), tvec.copy(),
        )
        rvec, tvec = rvec_r, tvec_r
    except cv2.error:
        pass

    R_near, _ = cv2.Rodrigues(rvec)
    t_near = tvec.flatten()

    proj, _ = cv2.projectPoints(obj_inl.reshape(-1, 3), rvec, tvec, K64, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - img_inl.reshape(-1, 2), axis=1)

    return {
        'R': R_near.astype(np.float64),
        't': t_near.astype(np.float64),
        'K': K64,
        'n_correspondences': int(len(pts_far)),
        'n_ray_hits': int(len(hit_idx)),
        'n_pnp_inliers': int(len(inl_idx)),
        'mean_reproj_err_px': float(err.mean()),
        'max_reproj_err_px': float(err.max()),
        'ransac_reproj_err_px': float(ransac_reproj_err_px),
        'hit_idx': hit_idx.tolist(),
        'pnp_inlier_idx_in_hits': inl_idx,
    }
