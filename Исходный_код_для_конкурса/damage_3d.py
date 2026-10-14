
from __future__ import annotations

import json
import struct
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from diag import dlog, dlog_arr, dlog_cv2_imread, dlog_timer


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(name: str) -> str:
    import re
    cleaned = re.sub(r'[^\w\-. ]+', '_', str(name), flags=re.UNICODE).strip()
    cleaned = cleaned.replace('/', '_').replace('\\', '_')
    return cleaned[:180] if cleaned else 'item'


def find_damaged_parts(
    damage_masks: List[Dict],
    part_masks: List[Dict],
    min_overlap_px: int = 20,
) -> List[Dict]:
    results = []
    skipped_small = 0
    for di, dm in enumerate(damage_masks):
        for pm in part_masks:
            overlap = int(np.logical_and(dm['mask'], pm['mask']).sum())
            if overlap >= min_overlap_px:
                results.append({
                    'damage_idx': di,
                    'part': pm['label'],
                    'damage_mask': dm['mask'],
                    'part_mask': pm['mask'],
                    'overlap_px': overlap,
                })
            elif overlap > 0:
                skipped_small += 1
    if skipped_small:
        print(f'  Пересечения ниже порога min_overlap_px={min_overlap_px}: {skipped_small}')
    dlog(
        f'[find_damaged_parts] hits={len(results)} skipped_small_overlap={skipped_small} '
        f'min_overlap_px={min_overlap_px}'
    )
    return results


def build_union_mask(mask_items: List[Dict]) -> np.ndarray:
    if not mask_items:
        raise ValueError('Пустой список масок деталей')
    out = np.zeros_like(mask_items[0]['mask'], dtype=np.uint8)
    for item in mask_items:
        out = np.logical_or(out, item['mask'] > 0)
    return out.astype(np.uint8)


def union_masks_for_label(mask_items: List[Dict], label: str) -> Optional[np.ndarray]:
    out = None
    for item in mask_items:
        if item.get('label') != label:
            continue
        m = np.asarray(item['mask']) > 0
        out = m.copy() if out is None else np.logical_or(out, m)
    return None if out is None else out.astype(np.uint8)


GLB_COMP = {
    5120: ('b', 1), 5121: ('B', 1), 5122: ('h', 2), 5123: ('H', 2),
    5125: ('I', 4), 5126: ('f', 4),
}
GLB_TYPE_N = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT2': 4, 'MAT3': 9, 'MAT4': 16}


def norm_part_key(name: str) -> str:
    return str(name).strip().lower()


def load_glb_chunks(path: Path):
    raw = Path(path).read_bytes()
    offset = 12
    j_data = None
    bin_data = None
    while offset < len(raw):
        chunk_len, chunk_type = struct.unpack_from('<II', raw, offset)
        offset += 8
        chunk_data = raw[offset:offset + chunk_len]
        offset += chunk_len
        if chunk_type == 0x4E4F534A:
            j_data = json.loads(chunk_data.decode('utf-8'))
        elif chunk_type == 0x004E4942:
            bin_data = chunk_data
    if j_data is None or bin_data is None:
        raise RuntimeError('GLB прочитан некорректно: нет JSON или BIN chunk')
    return j_data, bin_data


def _read_accessor(j_data, bin_data, idx) -> np.ndarray:
    acc = j_data['accessors'][idx]
    bv = j_data['bufferViews'][acc['bufferView']]
    base = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
    count = acc['count']
    fmt, comp_size = GLB_COMP[acc['componentType']]
    n_comp = GLB_TYPE_N[acc['type']]
    stride = bv.get('byteStride', comp_size * n_comp)
    rows = []
    for i in range(count):
        row = [
            struct.unpack_from('<' + fmt, bin_data,
                               base + i * stride + j * comp_size)[0]
            for j in range(n_comp)
        ]
        rows.append(row if n_comp > 1 else row[0])
    return np.array(rows)


def _node_matrix(node):
    M = np.eye(4, dtype=np.float64)
    if 'matrix' in node and node['matrix'] is not None:
        M = np.array(node['matrix'], dtype=np.float64).reshape(4, 4).T
    else:
        if node.get('translation') is not None:
            M[:3, 3] = node['translation']
        if node.get('rotation') is not None:
            qx, qy, qz, qw = node['rotation']
            M[:3, :3] = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ], dtype=np.float64)
        if node.get('scale') is not None:
            M[:3, :3] *= np.array(node['scale'], dtype=np.float64)
    return M


def _build_parent_map(nodes):
    parent = {i: None for i in range(len(nodes))}
    for i, n in enumerate(nodes):
        for ch in n.get('children') or []:
            parent[ch] = i
    return parent


def _world_matrix(nodes, parent_map, idx):
    M = np.eye(4, dtype=np.float64)
    chain = []
    cur = idx
    while cur is not None:
        chain.append(cur)
        cur = parent_map[cur]
    for k in reversed(chain):
        M = M @ _node_matrix(nodes[k])
    return M


def _choose_up_axis_auto(verts: np.ndarray) -> str:
    ext = verts.max(0) - verts.min(0)
    idx = int(np.argmin(ext))
    axis = ['x', 'y', 'z'][idx]
    dlog(
        f'[_choose_up_axis_auto] extents XYZ={ext.tolist()} argmin_idx={idx} -> '
        f'chosen_up_axis={axis}'
    )
    return axis


def _reorient_to_y_up(verts: np.ndarray, up_axis='auto', flip_up=False):
    axis = str(up_axis).strip().lower()
    if axis not in ('auto', 'x', 'y', 'z'):
        raise ValueError(f'Некорректный up_axis={up_axis}')
    if axis == 'auto':
        axis = _choose_up_axis_auto(verts)

    out = verts.astype(np.float32).copy()
    if axis == 'x':
        out = out[:, [1, 0, 2]]
    elif axis == 'z':
        out = out[:, [0, 2, 1]]
    if flip_up:
        out[:, 1] *= -1.0
    return out, {'chosen_up_axis': axis, 'flip_up': bool(flip_up)}


def load_glb(path, up_axis='auto', flip_up=False, debug_axes=False, apply_reorient=True):
    j_data, bin_data = load_glb_chunks(Path(path))
    nodes = j_data.get('nodes', [])
    all_v, all_f = [], []
    counter = [0]
    part_acc = defaultdict(lambda: {'v': [], 'f': [], 'off': 0})

    def append_mesh(part_name: str, v_world, f_tri):
        v = np.asarray(v_world, dtype=np.float32)
        f_tri = np.asarray(f_tri, dtype=np.int32)
        all_v.append(v)
        all_f.append(f_tri + counter[0])
        counter[0] += len(v)
        pa = part_acc[part_name]
        pa['f'].append(f_tri + pa['off'])
        pa['v'].append(v)
        pa['off'] += len(v)

    def traverse(node_idx, parent_M):
        node = nodes[node_idx]
        M = parent_M @ _node_matrix(node)
        raw_name = node.get('name')
        part_name = str(raw_name).strip() if raw_name and str(raw_name).strip() else '__unnamed__'

        if 'mesh' in node:
            mesh = j_data['meshes'][node['mesh']]
            for prim in mesh.get('primitives', []):
                if 'POSITION' not in prim.get('attributes', {}):
                    continue
                if prim.get('mode', 4) != 4:
                    continue
                v_local = _read_accessor(j_data, bin_data, prim['attributes']['POSITION']).astype(np.float64)
                v_h = np.hstack([v_local, np.ones((len(v_local), 1), dtype=np.float64)])
                v_world = (M @ v_h.T).T[:, :3]
                if 'indices' in prim:
                    f = _read_accessor(j_data, bin_data, prim['indices']).astype(np.int32)
                    if f.ndim == 1:
                        if len(f) % 3 != 0:
                            raise ValueError('Индексы не кратны 3')
                        f = f.reshape(-1, 3)
                else:
                    if len(v_world) % 3 != 0:
                        raise ValueError('Вершины не кратны 3')
                    f = np.arange(len(v_world), dtype=np.int32).reshape(-1, 3)
                append_mesh(part_name, v_world, f)

        for child_idx in node.get('children', []):
            traverse(child_idx, M)

    scenes = j_data.get('scenes', [])
    root_nodes = scenes[0]['nodes'] if scenes else list(range(len(nodes)))
    for ni in root_nodes:
        traverse(ni, np.eye(4, dtype=np.float64))

    if not all_v:
        raise ValueError('Нет геометрии в GLB')

    verts = np.concatenate(all_v, axis=0)
    faces = np.concatenate(all_f, axis=0)

    part_meshes = {}
    for pname, acc in part_acc.items():
        if not acc['v']:
            continue
        part_meshes[pname] = (
            np.concatenate(acc['v'], axis=0),
            np.concatenate(acc['f'], axis=0),
        )

    if apply_reorient:
        verts, ax_info = _reorient_to_y_up(verts, up_axis=up_axis, flip_up=flip_up)
        part_meshes_rot = {}
        for pname, (pv, pf) in part_meshes.items():
            pv2, _ = _reorient_to_y_up(pv, up_axis=ax_info['chosen_up_axis'], flip_up=flip_up)
            part_meshes_rot[pname] = (pv2, pf)
        part_meshes = part_meshes_rot

    print(f'  Загружено full mesh: {len(verts)} вершин, {len(faces)} граней')
    named = [k for k in part_meshes if k != '__unnamed__']
    print(f'  Именованных деталей в GLB: {len(named)}')
    return verts, faces, part_meshes


def get_keypoint_world_positions(j_data, kp_node_names: List[str]) -> Dict[str, np.ndarray]:
    nodes = j_data['nodes']
    parent_map = _build_parent_map(nodes)
    name_to_idx = {(n.get('name') or ''): i for i, n in enumerate(nodes)}
    result = {}
    for nm in kp_node_names:
        idx = name_to_idx.get(nm)
        if idx is None:
            continue
        M = _world_matrix(nodes, parent_map, idx)
        result[nm] = M[:3, 3].copy()
    return result


def get_part_mesh_world(j_data, bin_data, part_node_name: str):
    nodes = j_data['nodes']
    parent_map = _build_parent_map(nodes)
    name_to_idx = {(n.get('name') or ''): i for i, n in enumerate(nodes)}
    idx = name_to_idx.get(part_node_name)
    if idx is None:
        return None
    node = nodes[idx]
    if 'mesh' not in node:
        return None
    M = _world_matrix(nodes, parent_map, idx)
    mesh = j_data['meshes'][node['mesh']]
    all_v, all_f = [], []
    voff = 0
    for prim in mesh.get('primitives', []):
        if 'POSITION' not in prim.get('attributes', {}):
            continue
        if prim.get('mode', 4) != 4:
            continue
        v_local = _read_accessor(j_data, bin_data, prim['attributes']['POSITION']).astype(np.float64)
        v_h = np.hstack([v_local, np.ones((len(v_local), 1))])
        v_world = (M @ v_h.T).T[:, :3]
        if 'indices' in prim:
            f = _read_accessor(j_data, bin_data, prim['indices']).astype(np.int32).reshape(-1, 3)
        else:
            f = np.arange(len(v_world), dtype=np.int32).reshape(-1, 3)
        all_v.append(v_world)
        all_f.append(f + voff)
        voff += len(v_world)
    if not all_v:
        return None
    return np.concatenate(all_v, axis=0), np.concatenate(all_f, axis=0)


def simplify_mesh(verts, faces, max_faces=None, tag='mesh', quiet=False):
    if max_faces is None or len(faces) <= max_faces:
        if not quiet:
            print(f'  {tag}: без децимации, {len(verts)} вершин, {len(faces)} граней')
        return verts.copy(), faces.copy()

    step = max(1, len(faces) // max_faces)
    faces_s = faces[::step]
    used = np.unique(faces_s)
    remap = np.full(len(verts), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    verts_s = verts[used]
    faces_s = remap[faces_s]
    faces_s = faces_s[(faces_s >= 0).all(axis=1)]
    if not quiet:
        print(f'  {tag}: децимация x{step} -> {len(verts_s)} вершин, {len(faces_s)} граней')
    return verts_s, faces_s


def merge_part_meshes(part_meshes: dict, names: list):
    names = [n for n in names if n in part_meshes and n != '__unnamed__']
    if not names:
        return None, None
    chunks_v, chunks_f = [], []
    offset = 0
    for n in names:
        v, f = part_meshes[n]
        v = np.asarray(v, dtype=np.float32)
        f = np.asarray(f, dtype=np.int32)
        if len(f) == 0:
            continue
        chunks_v.append(v)
        chunks_f.append(f + offset)
        offset += len(v)
    if not chunks_v:
        return None, None
    return np.concatenate(chunks_v, axis=0), np.concatenate(chunks_f, axis=0)


def build_part_lookup(part_meshes: dict) -> dict:
    lookup = {}
    for k in part_meshes:
        if k == '__unnamed__':
            continue
        lookup[norm_part_key(k)] = k
    return lookup


def resolve_label_to_glb_key(label: str, lookup: dict, part_meshes: dict):
    if not label:
        return None
    if label in part_meshes and label != '__unnamed__':
        return label
    return lookup.get(norm_part_key(label))


def ray_cast(damage_mask, verts, faces, R, t, K, subsample=6):
    dlog(
        f'[ray_cast] START faces={len(faces)} verts={len(verts)} subsample={subsample} '
        f'mask_shape={damage_mask.shape}'
    )
    t0 = time.perf_counter()
    K_inv = np.linalg.inv(K)
    R_T = R.T
    cam_pos = (R_T @ (-t)).astype(np.float64)

    ys, xs = np.where(damage_mask > 0)
    if len(xs) == 0:
        return []

    idx = np.arange(0, len(xs), max(1, subsample))
    xs = xs[idx].astype(np.float64)
    ys = ys[idx].astype(np.float64)
    N = len(xs)

    dirs_cam = K_inv @ np.stack([xs, ys, np.ones(N, dtype=np.float64)])
    dirs = (R_T @ dirs_cam).T
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-10

    v0 = verts[faces[:, 0]].astype(np.float64)
    e1 = verts[faces[:, 1]].astype(np.float64) - v0
    e2 = verts[faces[:, 2]].astype(np.float64) - v0

    hits = []
    batch = 128
    print(f'  Ray casting: {N} лучей x {len(faces)} граней...')

    for start in range(0, N, batch):
        d = dirs[start:start + batch]
        B = len(d)
        h = np.cross(d[:, None, :], e2[None, :, :])
        a = np.einsum('fi,bfi->bf', e1, h)
        ok = np.abs(a) > 1e-8
        inv = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
        s = cam_pos - v0
        u = inv * np.einsum('fi,bfi->bf', s, h)
        q = np.cross(s[None, :, :], e1[None, :, :])
        v = inv * np.einsum('bi,bfi->bf', d, q)
        t_hit = inv * np.einsum('fi,bfi->bf', e2, q)
        valid = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t_hit > 1e-6)
        for bi in range(B):
            row = valid[bi]
            if not row.any():
                continue
            t_vals = np.where(row, t_hit[bi], np.inf)
            fi = int(t_vals.argmin())
            pt = cam_pos + t_vals[fi] * dirs[start + bi]
            hits.append((fi, pt))

    print(f'  Попаданий: {len(hits)} / {N}')
    dlog(
        f'[ray_cast] END hits={len(hits)}/{N} time_ms={(time.perf_counter() - t0) * 1000:.1f}'
    )
    return hits


def compute_severity(hits, verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total_area = float(areas.sum())
    damaged_fi = set(fi for fi, _ in hits)
    damage_area = float(sum(areas[fi] for fi in damaged_fi))
    severity = min(damage_area / (total_area + 1e-8), 1.0)
    return severity, damage_area, total_area


def render_mesh_debug_view(verts, faces, R, t, K, img_h, img_w, highlight_faces=None):
    cam = (R @ verts.T).T + t
    front = cam[:, 2] > 1e-6
    uv = (K @ cam.T).T
    px = (uv[:, :2] / (uv[:, 2:3] + 1e-10)).astype(np.float32)

    canvas = np.full((img_h, img_w, 3), 245, dtype=np.uint8)
    highlight_faces = set(highlight_faces or [])

    face_depth = cam[faces][:, :, 2].mean(axis=1)
    order = np.argsort(face_depth)[::-1]
    light_dir = np.array([0.35, -0.25, 1.0], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir) + 1e-10

    for fi in order:
        face = faces[fi]
        if not (front[face[0]] and front[face[1]] and front[face[2]]):
            continue
        tri_cam = cam[face]
        normal = np.cross(tri_cam[1] - tri_cam[0], tri_cam[2] - tri_cam[0])
        n_norm = np.linalg.norm(normal)
        if n_norm < 1e-10:
            continue
        normal = normal / n_norm
        if normal[2] >= 0:
            continue
        intensity = max(0.20, min(1.0, -float(np.dot(normal, light_dir))))
        if fi in highlight_faces:
            base = np.array([45, 70, 225], dtype=np.float32)
            color = np.clip(base * (0.55 + 0.45 * intensity), 0, 255).astype(np.uint8)
            edge_color = (35, 45, 150)
            thickness = 1
        else:
            base = np.array([195, 210, 220], dtype=np.float32)
            color = np.clip(base * (0.55 + 0.45 * intensity), 0, 255).astype(np.uint8)
            edge_color = (170, 185, 195)
            thickness = 1

        pts = np.round(px[face]).astype(np.int32)
        if (np.any(pts[:, 0] < -img_w) or np.any(pts[:, 0] > 2 * img_w)
                or np.any(pts[:, 1] < -img_h) or np.any(pts[:, 1] > 2 * img_h)):
            continue
        cv2.fillConvexPoly(canvas, pts, tuple(int(x) for x in color.tolist()), lineType=cv2.LINE_AA)
        if fi in highlight_faces:
            cv2.polylines(canvas, [pts], True, edge_color, thickness, lineType=cv2.LINE_AA)

    return canvas


def overlay_projected_hit_points(img, hits, R, t, K, color=(40, 60, 230), radius=4):
    out = img.copy()
    for _, hp in hits:
        cp = R @ hp.astype(np.float32) + t
        if cp[2] <= 0:
            continue
        uv = K @ cp
        px_x = int(uv[0] / uv[2])
        px_y = int(uv[1] / uv[2])
        if 0 <= px_x < out.shape[1] and 0 <= px_y < out.shape[0]:
            cv2.circle(out, (px_x, px_y), radius, color, -1, lineType=cv2.LINE_AA)
    return out


def visualize_mesh_damage_debug(mesh_view_with_hits, severity, hit_count, out_path):
    vis = mesh_view_with_hits.copy()
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 58), (235, 235, 235), -1)
    cv2.putText(vis, f'3D projected damage, severity={severity*100:.2f}%', (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, lineType=cv2.LINE_AA)
    cv2.putText(vis, f'hit samples: {hit_count}', (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (70, 70, 70), 1, lineType=cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis)


def draw_severity_on_photo(img_bgr, damage_mask, hits, R, t, K, severity):
    vis = img_bgr.copy()
    h, w = vis.shape[:2]
    red = np.array([0, 0, 200], dtype=np.uint8)
    dm = damage_mask > 0
    vis[dm] = (vis[dm].astype(np.float32) * 0.4 + red.astype(np.float32) * 0.6).astype(np.uint8)
    for _, hp in hits:
        cp = R @ hp.astype(np.float32) + t
        if cp[2] > 0:
            uv = K @ cp
            px_x = int(uv[0] / uv[2])
            px_y = int(uv[1] / uv[2])
            if 0 <= px_x < w and 0 <= px_y < h:
                cv2.circle(vis, (px_x, px_y), 2, (0, 255, 0), -1)
    cv2.putText(vis, f'Part damage (3D area): {severity * 100:.2f}%', (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(vis, f'Ray hits: {len(hits)}', (15, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
    return vis


def _project_points(verts, R, t, K):
    cam = (R @ verts.T).T + t
    front = cam[:, 2] > 0
    uv = (K @ cam.T).T
    px = (uv[:, :2] / (uv[:, 2:3] + 1e-10)).astype(np.float32)
    return px, front, cam


def render_wireframe(
    verts, faces, R, t, K, img_h, img_w,
    bg=(245, 245, 248), line_color=(40, 40, 40), line_thickness=1,
):
    canvas = np.full((img_h, img_w, 3), bg, dtype=np.uint8)
    if len(verts) == 0 or len(faces) == 0:
        return canvas

    px, front, cam = _project_points(verts, R, t, K)
    faces = np.asarray(faces, dtype=np.int32)

    v0 = cam[faces[:, 0]]
    v1 = cam[faces[:, 1]]
    v2 = cam[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    facing = (normals * -v0).sum(axis=1) > 0
    visible = facing & front[faces].all(axis=1)

    depth = cam[faces][:, :, 2].mean(axis=1)
    order = np.argsort(-depth)

    px_int = px.astype(np.int32)
    h, w = img_h, img_w
    for fi in order:
        if not visible[fi]:
            continue
        i0, i1, i2 = faces[fi]
        p0, p1, p2 = px_int[i0], px_int[i1], px_int[i2]
        for a, b in ((p0, p1), (p1, p2), (p2, p0)):
            if (0 <= a[0] < w or 0 <= b[0] < w) and (0 <= a[1] < h or 0 <= b[1] < h):
                cv2.line(canvas, tuple(a), tuple(b), line_color, line_thickness, cv2.LINE_AA)
    return canvas


def overlay_mesh_on_photo(
    img, verts, faces, R, t, K,
    line_color=(0, 255, 140), fill_color=(0, 255, 140),
    alpha_fill=0.25, line_thickness=1,
):
    h, w = img.shape[:2]
    out = img.copy()
    if len(verts) == 0 or len(faces) == 0:
        return out

    px, front, cam = _project_points(verts, R, t, K)
    faces = np.asarray(faces, dtype=np.int32)
    v0 = cam[faces[:, 0]]
    v1 = cam[faces[:, 1]]
    v2 = cam[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    facing = (normals * -v0).sum(axis=1) > 0
    visible = facing & front[faces].all(axis=1)

    sil = np.zeros((h, w), dtype=np.uint8)
    tris_for_fill = []
    px_int = px.astype(np.int32)
    for fi in np.where(visible)[0]:
        i0, i1, i2 = faces[fi]
        tris_for_fill.append(np.array([[px_int[i0]], [px_int[i1]], [px_int[i2]]], dtype=np.int32))
    if tris_for_fill:
        cv2.fillPoly(sil, tris_for_fill, 1)

    mask_bool = sil > 0
    if mask_bool.any():
        col = np.array(fill_color, dtype=np.float32)
        out[mask_bool] = (col * alpha_fill + out[mask_bool].astype(np.float32) * (1 - alpha_fill)).astype(np.uint8)

    depth = cam[faces][:, :, 2].mean(axis=1)
    order = np.argsort(-depth)
    for fi in order:
        if not visible[fi]:
            continue
        i0, i1, i2 = faces[fi]
        p0, p1, p2 = px_int[i0], px_int[i1], px_int[i2]
        cv2.line(out, tuple(p0), tuple(p1), line_color, line_thickness, cv2.LINE_AA)
        cv2.line(out, tuple(p1), tuple(p2), line_color, line_thickness, cv2.LINE_AA)
        cv2.line(out, tuple(p2), tuple(p0), line_color, line_thickness, cv2.LINE_AA)
    return out


def annotate_keypoints(img_bgr, image_pts, projected_pts, names, inlier_mask=None):
    vis = img_bgr.copy()
    n = len(image_pts)
    if inlier_mask is None:
        inlier_mask = [True] * n

    for (x_obs, y_obs), (x_proj, y_proj) in zip(image_pts, projected_pts):
        cv2.line(vis,
                 (int(round(x_obs)), int(round(y_obs))),
                 (int(round(x_proj)), int(round(y_proj))),
                 (255, 255, 255), 1, lineType=cv2.LINE_AA)

    for (x, y), nm, is_in in zip(image_pts, names, inlier_mask):
        xi, yi = int(round(x)), int(round(y))
        fill_color = (0, 0, 255) if is_in else (0, 140, 255)
        cv2.circle(vis, (xi, yi), 7, fill_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, (xi, yi), 8, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        short = nm.split('_')[-1]
        label = short if is_in else f'{short}!'
        cv2.putText(vis, label, (xi + 8, yi - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (xi + 8, yi - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, fill_color, 1, cv2.LINE_AA)

    for (x, y) in projected_pts:
        xi, yi = int(round(x)), int(round(y))
        cv2.drawMarker(vis, (xi, yi), (0, 255, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
    return vis


def make_camera_matrix(width: int, height: int, focal: Optional[float] = None):
    if focal is None or focal <= 0:
        focal = 0.72 * float(width)
    cx, cy = width / 2.0, height / 2.0
    K = np.array([
        [focal, 0.0, cx],
        [0.0, focal, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return K, np.zeros(5, dtype=np.float64)


def _read_focal_exif(image_path):
    try:
        from PIL import Image, ExifTags
    except Exception as exc:
        dlog(f'[K/exif] Pillow недоступен: {exc}')
        return None, None, None, None
    try:
        img = Image.open(str(image_path))
        exif = img.getexif()
    except Exception as exc:
        dlog(f'[K/exif] не удалось открыть {image_path}: {exc}')
        return None, None, None, None
    if not exif:
        return None, None, None, None

    tag_id_to_name = ExifTags.TAGS
    focal_35mm = None
    focal_mm = None
    make = None
    model = None

    for k, v in exif.items():
        tag = tag_id_to_name.get(k, k)
        if tag == 'Make':
            try:
                make = str(v).strip()
            except Exception:
                pass
        elif tag == 'Model':
            try:
                model = str(v).strip()
            except Exception:
                pass

    try:
        exif_ifd = exif.get_ifd(0x8769)
    except Exception:
        exif_ifd = {}

    def _to_float(x):
        try:
            return float(x)
        except Exception:
            try:
                return float(x.numerator) / float(x.denominator)
            except Exception:
                return None

    for k, v in exif_ifd.items():
        tag = tag_id_to_name.get(k, k)
        if tag == 'FocalLengthIn35mmFilm':
            f = _to_float(v)
            if f and f > 0:
                focal_35mm = f
        elif tag == 'FocalLength':
            f = _to_float(v)
            if f and f > 0:
                focal_mm = f

    return focal_35mm, focal_mm, make, model


def try_solve_pnp(
    obj_pts,
    img_pts,
    K,
    dist,
    ransac_reproj_err_px: float = 8.0,
    ransac_min_inliers: int = 4,
    use_ransac: bool = True,
):
    methods = []
    if hasattr(cv2, 'SOLVEPNP_SQPNP'):
        methods.append(('SQPNP', cv2.SOLVEPNP_SQPNP))
    methods.append(('ITERATIVE', cv2.SOLVEPNP_ITERATIVE))
    if obj_pts.shape[0] >= 6:
        methods.append(('EPNP', cv2.SOLVEPNP_EPNP))

    n_total = int(obj_pts.shape[0])
    op = obj_pts.reshape(-1, 1, 3).astype(np.float64)
    ip = img_pts.reshape(-1, 1, 2).astype(np.float64)

    results = {}
    for name, flag in methods:
        try:
            if use_ransac:
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    op, ip, K, dist,
                    flags=flag,
                    reprojectionError=float(ransac_reproj_err_px),
                    iterationsCount=500,
                    confidence=0.999,
                )
                if not ok or inliers is None:
                    results[name] = {'ok': False, 'reason': 'solvePnPRansac returned False'}
                    continue
                inlier_idx = inliers.flatten().astype(int).tolist()
                if len(inlier_idx) < ransac_min_inliers:
                    results[name] = {
                        'ok': False,
                        'reason': f'inliers={len(inlier_idx)} < {ransac_min_inliers}',
                    }
                    continue
                op_in = op[inlier_idx]
                ip_in = ip[inlier_idx]
                try:
                    rvec_r, tvec_r = cv2.solvePnPRefineLM(
                        op_in, ip_in, K, dist, rvec.copy(), tvec.copy(),
                    )
                    rvec_f, tvec_f = rvec_r, tvec_r
                except cv2.error:
                    rvec_f, tvec_f = rvec, tvec
            else:
                ok, rvec, tvec = cv2.solvePnP(op, ip, K, dist, flags=flag)
                if not ok:
                    results[name] = {'ok': False, 'reason': 'solvePnP returned False'}
                    continue
                inlier_idx = list(range(n_total))
                rvec_f, tvec_f = rvec, tvec

            proj_all, _ = cv2.projectPoints(obj_pts, rvec_f, tvec_f, K, dist)
            proj_all = proj_all.reshape(-1, 2)
            err_all = np.linalg.norm(proj_all - img_pts, axis=1)

            inlier_mask = np.zeros(n_total, dtype=bool)
            inlier_mask[inlier_idx] = True
            outlier_idx = np.where(~inlier_mask)[0].tolist()
            err_in = err_all[inlier_mask]

            results[name] = {
                'ok': True,
                'rvec': rvec_f.flatten().tolist(),
                'tvec': tvec_f.flatten().tolist(),
                'mean_err_px': float(err_in.mean()),
                'max_err_px': float(err_in.max()),
                'per_point_err': err_all.tolist(),
                'projected': proj_all.tolist(),
                'rvec_np': rvec_f,
                'tvec_np': tvec_f,
                'inlier_idx': inlier_idx,
                'outlier_idx': outlier_idx,
                'n_inliers': int(inlier_mask.sum()),
                'n_outliers': int(n_total - inlier_mask.sum()),
                'n_total': n_total,
                'ransac_reproj_err_px': float(ransac_reproj_err_px) if use_ransac else None,
                'used_ransac': bool(use_ransac),
            }
        except cv2.error as e:
            results[name] = {'ok': False, 'reason': str(e)}
    return results


def build_correspondences_from_yolo_keypoints(
    yolo_keypoints: List[Dict],
    kp_xyz: Dict[str, np.ndarray],
):
    obj, img, names = [], [], []
    for kp in yolo_keypoints:
        glb_name = kp.get('glb_name')
        if not glb_name or glb_name not in kp_xyz:
            continue
        obj.append(kp_xyz[glb_name])
        img.append([kp['cx'], kp['cy']])
        names.append(glb_name)
    if len(obj) < 4:
        raise RuntimeError(f'Слишком мало корреспонденций для PnP: {len(obj)} < 4')
    return np.asarray(obj, dtype=np.float64), np.asarray(img, dtype=np.float64), names


def make_parts_overlay(img_bgr: np.ndarray, part_masks: List[Dict]) -> np.ndarray:
    if img_bgr is None:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    h, w = img_bgr.shape[:2]
    overlay = img_bgr.copy()
    palette = [
        (255, 100, 0), (0, 200, 255), (255, 60, 200), (60, 255, 100),
        (200, 200, 0), (200, 100, 255), (255, 180, 80), (80, 180, 255),
        (255, 80, 80), (160, 220, 100), (100, 160, 255), (220, 80, 160),
    ]
    for i, item in enumerate(part_masks):
        m = np.asarray(item['mask']) > 0
        if not m.any():
            continue
        color = np.array(palette[i % len(palette)], dtype=np.float32)
        overlay[m] = (overlay[m].astype(np.float32) * 0.55 + color * 0.45).astype(np.uint8)

        ys, xs = np.where(m)
        cx, cy = int(xs.mean()), int(ys.mean())
        cv2.putText(overlay, item['label'], (max(0, cx - 60), max(20, cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, item['label'], (max(0, cx - 60), max(20, cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return overlay


def make_damages_overlay(img_bgr: np.ndarray, damage_masks: List[Dict]) -> np.ndarray:
    if img_bgr is None:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    overlay = img_bgr.copy()
    red = np.array([0, 0, 220], dtype=np.float32)
    for item in damage_masks:
        m = np.asarray(item['mask']) > 0
        if not m.any():
            continue
        overlay[m] = (overlay[m].astype(np.float32) * 0.4 + red * 0.6).astype(np.uint8)
    cv2.putText(overlay, f'damages: {len(damage_masks)}', (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 220), 2, cv2.LINE_AA)
    return overlay


def make_keypoints_visualization(img_bgr: np.ndarray, yolo_keypoints: List[Dict]) -> np.ndarray:
    vis = img_bgr.copy()
    palette = {
        'bb': (0, 200, 255), 'tr': (255, 100, 0), 'fl_l': (0, 255, 0),
        'fl_r': (0, 255, 150), 'lb': (200, 0, 255), 'bw': (0, 100, 255),
    }
    for kp in yolo_keypoints:
        prefix = kp.get('prefix') or ''
        color = palette.get(prefix, (200, 200, 200))
        x = int(round(kp['cx'])); y = int(round(kp['cy']))
        cv2.circle(vis, (x, y), 8, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, (x, y), 9, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        label = kp.get('glb_name') or f"{prefix}_{kp.get('k')}"
        short = label.split('_')[-1] if '_' in label else label
        cv2.putText(vis, short, (x + 10, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, short, (x + 10, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis
