#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import html
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

FRAME_OFFSETS_S = (-90.0, -60.0, -30.0, 0.0)
DEFAULT_DPI = 120
DEFAULT_SIZE_PX = 768


# ============================================================
# I/O
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Render native benchmark topdown/radar images and gallery pages (v3).')
    p.add_argument('--root', type=str, required=True)
    p.add_argument('--source-jsonl', type=str, default='release/all/all_release_samples.jsonl')
    p.add_argument('--out-dir', type=str, default='release/images')
    p.add_argument('--gallery-dir', type=str, default='release/galleries')
    p.add_argument('--index-jsonl', type=str, default='release/manifests/rendered_image_index.jsonl')
    p.add_argument('--report-json', type=str, default='reports/release_reports/render_native_benchmark_images_report.json')
    p.add_argument('--max-samples', type=int, default=0)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--size-px', type=int, default=DEFAULT_SIZE_PX)
    p.add_argument('--dpi', type=int, default=DEFAULT_DPI)
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


# ============================================================
# Scene helpers
# ============================================================

def get_scene_spec(row: Dict[str, Any]) -> Dict[str, Any]:
    if 'scene_spec' in row and isinstance(row['scene_spec'], dict):
        return row['scene_spec']
    inputs = row.get('inputs', {})
    if isinstance(inputs, dict) and isinstance(inputs.get('scene_spec'), dict):
        return inputs['scene_spec']
    raise KeyError(f"row {row.get('sample_id')} has no scene_spec")


def vessel_id(v: Dict[str, Any]) -> str:
    return str(v.get('id') or v.get('vessel_id') or v.get('target_id') or 'v')


def vessel_length(v: Dict[str, Any]) -> float:
    return float(v.get('length_m') or 120.0)


def heading_to_vec(heading_deg: float) -> np.ndarray:
    rad = math.radians(float(heading_deg))
    return np.array([math.sin(rad), math.cos(rad)], dtype=float)


def move_state(v: Dict[str, Any], dt_s: float) -> Dict[str, Any]:
    x = float(v.get('x_m', 0.0)) + float(v.get('speed_mps', 0.0)) * dt_s * math.sin(math.radians(float(v.get('heading_deg', 0.0))))
    y = float(v.get('y_m', 0.0)) + float(v.get('speed_mps', 0.0)) * dt_s * math.cos(math.radians(float(v.get('heading_deg', 0.0))))
    out = dict(v)
    out['x_m'] = x
    out['y_m'] = y
    return out


def scene_states_at_offset(scene: Dict[str, Any], offset_s: float) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    own = move_state(scene['ownship'], offset_s)
    tgts = [move_state(t, offset_s) for t in scene.get('targets', [])]
    return own, tgts


def ship_polygon(x: float, y: float, heading_deg: float, length_m: float, width_m: float) -> Tuple[np.ndarray, np.ndarray]:
    L = float(length_m)
    W = float(width_m)
    rect = np.array([[W / 2, L / 2], [-W / 2, L / 2], [-W / 2, -L / 2], [W / 2, -L / 2]], dtype=float)
    tri = np.array([[0.0, L / 2 + 0.35 * L], [W / 2, L / 2], [-W / 2, L / 2]], dtype=float)
    ang = math.radians(float(heading_deg))
    u = np.array([math.sin(ang), math.cos(ang)], dtype=float)
    r = np.array([u[1], -u[0]], dtype=float)
    R = np.stack([r, u], axis=1)
    return rect @ R.T + np.array([x, y]), tri @ R.T + np.array([x, y])


def relative_bearing_deg(own_heading_deg: float, rel_xy: np.ndarray) -> float:
    ang = math.degrees(math.atan2(rel_xy[0], rel_xy[1]))
    rel = ang - float(own_heading_deg)
    while rel <= -180.0:
        rel += 360.0
    while rel > 180.0:
        rel -= 360.0
    return rel


def get_area_context(scene: Dict[str, Any]) -> Dict[str, Any]:
    area = scene.get('area_context')
    if isinstance(area, dict):
        return area
    return {}


def infer_area_mode(scene: Dict[str, Any]) -> str:
    area = get_area_context(scene)
    if area.get('channel_context'):
        return 'channel'
    if area.get('area_type') == 'tss' or area.get('tss_context'):
        return 'tss'
    if 'tss_lane_heading_deg' in scene:
        return 'legacy_tss'
    return 'open'


def infer_area_heading_deg(scene: Dict[str, Any], own: Dict[str, Any]) -> float:
    area = get_area_context(scene)
    mode = infer_area_mode(scene)
    if mode == 'channel':
        return float(area.get('channel_heading_deg', own.get('heading_deg', 0.0)))
    if mode == 'tss':
        return float(area.get('tss_lane_heading_deg', area.get('separation_zone_geometry', {}).get('heading_deg', own.get('heading_deg', 0.0))))
    if mode == 'legacy_tss':
        return float(scene.get('tss_lane_heading_deg', own.get('heading_deg', 0.0)))
    return float(own.get('heading_deg', 0.0))


# ============================================================
# View framing
# ============================================================

def oriented_rect_to_axis_ranges(center: np.ndarray, u: np.ndarray, r: np.ndarray, along_half: float, cross_half: float) -> Tuple[float, float]:
    corners = []
    for a in (-along_half, along_half):
        for c in (-cross_half, cross_half):
            p = center + u * a + r * c
            corners.append(p)
    x_half = max(abs(float(p[0] - center[0])) for p in corners)
    y_half = max(abs(float(p[1] - center[1])) for p in corners)
    return x_half, y_half


def dynamic_view(scene: Dict[str, Any]) -> Dict[str, Any]:
    own = scene['ownship']
    center = np.array([float(own.get('x_m', 0.0)), float(own.get('y_m', 0.0))], dtype=float)
    tgts = scene.get('targets', [])
    pts = [center] + [np.array([float(t.get('x_m', 0.0)), float(t.get('y_m', 0.0))], dtype=float) for t in tgts]
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    dx = max(abs(x - center[0]) for x in xs) if xs else 0.0
    dy = max(abs(y - center[1]) for y in ys) if ys else 0.0

    mode = infer_area_mode(scene)
    area = get_area_context(scene)
    heading_deg = infer_area_heading_deg(scene, own)
    ang = math.radians(heading_deg)
    u = np.array([math.sin(ang), math.cos(ang)], dtype=float)
    r = np.array([u[1], -u[0]], dtype=float)

    rels = [p - center for p in pts]
    along_vals = [float(np.dot(v, u)) for v in rels]
    cross_vals = [float(np.dot(v, r)) for v in rels]
    along_span = max(abs(v) for v in along_vals) if along_vals else 0.0
    cross_span = max(abs(v) for v in cross_vals) if cross_vals else 0.0

    if mode == 'channel':
        ch_w = float(area.get('channel_width_m', 350.0))
        along_half = max(900.0, along_span * 1.35 + 120.0)
        cross_half = max(ch_w * 0.95, cross_span * 1.40 + 90.0)
        x_half, y_half = oriented_rect_to_axis_ranges(center, u, r, along_half, cross_half)
        return {
            'mode': mode,
            'center': center,
            'x_half': min(2800.0, max(700.0, x_half * 1.05)),
            'y_half': min(2200.0, max(500.0, y_half * 1.05)),
            'u': u,
            'r': r,
            'along_half': along_half,
            'cross_half': cross_half,
            'heading_deg': heading_deg,
        }

    if mode in {'tss', 'legacy_tss'}:
        lane_w = float(area.get('lane_width_m', 350.0)) if mode == 'tss' else 350.0
        half_total = float(area.get('lane_half_total_width_m', lane_w * 1.5)) if mode == 'tss' else lane_w * 1.5
        along_half = max(1000.0, along_span * 1.35 + 150.0)
        cross_half = max(half_total * 1.15, cross_span * 1.35 + 110.0)
        x_half, y_half = oriented_rect_to_axis_ranges(center, u, r, along_half, cross_half)
        return {
            'mode': mode,
            'center': center,
            'x_half': min(3200.0, max(800.0, x_half * 1.05)),
            'y_half': min(2600.0, max(550.0, y_half * 1.05)),
            'u': u,
            'r': r,
            'along_half': along_half,
            'cross_half': cross_half,
            'heading_deg': heading_deg,
        }

    x_half = max(1200.0, min(4500.0, dx * 1.35 + 120.0))
    y_half = max(1200.0, min(4500.0, dy * 1.35 + 120.0))
    return {
        'mode': mode,
        'center': center,
        'x_half': x_half,
        'y_half': y_half,
        'u': u,
        'r': r,
        'along_half': max(x_half, y_half),
        'cross_half': min(x_half, y_half),
        'heading_deg': heading_deg,
    }


def display_ship_dims(ent: Dict[str, Any], view: Dict[str, Any]) -> Tuple[float, float]:
    true_len = vessel_length(ent)
    window_ref = math.sqrt((2.0 * view['x_half']) * (2.0 * view['y_half']))
    disp_len = min(max(true_len * 0.70, window_ref * 0.035), window_ref * 0.065)
    disp_wid = max(window_ref * 0.008, disp_len * 0.16)
    return disp_len, disp_wid


# ============================================================
# Overlay rendering
# ============================================================

def project_point_to_line(point: np.ndarray, line_point: np.ndarray, u: np.ndarray) -> np.ndarray:
    return line_point + u * float(np.dot(point - line_point, u))


def draw_local_parallel_lines(
    ax: plt.Axes,
    *,
    center: np.ndarray,
    u: np.ndarray,
    r: np.ndarray,
    along_half: float,
    offsets: List[float],
    color: str,
    center_dash_index: Optional[int] = None,
    linewidth_main: float = 1.0,
    linewidth_center: float = 0.8,
    alpha_main: float = 0.35,
    alpha_center: float = 0.28,
) -> None:
    L = along_half * 1.08
    for idx, off in enumerate(offsets):
        a = center + r * off - u * L
        b = center + r * off + u * L
        is_center = center_dash_index is not None and idx == center_dash_index
        ax.plot(
            [a[0], b[0]], [a[1], b[1]],
            linestyle='--' if is_center else '-',
            linewidth=linewidth_center if is_center else linewidth_main,
            alpha=alpha_center if is_center else alpha_main,
            color=color,
            zorder=0,
        )


def draw_tss(ax: plt.Axes, scene: Dict[str, Any], view: Dict[str, Any]) -> None:
    area = get_area_context(scene)
    center = view['center']
    u = view['u']
    r = view['r']
    lane_w = float(area.get('lane_width_m', 350.0))
    lane_n = int(area.get('lane_count', 3))
    half_total = float(area.get('lane_half_total_width_m', lane_w * max(1.5, lane_n / 2.0)))

    sep = area.get('separation_zone_geometry', {})
    anchor = np.array(sep.get('center', center.tolist()), dtype=float)
    local_center = project_point_to_line(center, anchor, u)

    if lane_n <= 1:
        offsets = [-half_total, 0.0, half_total]
        center_idx = 1
    else:
        offsets = list(np.linspace(-half_total, half_total, lane_n + 1))
        center_idx = len(offsets) // 2

    draw_local_parallel_lines(
        ax,
        center=local_center,
        u=u,
        r=r,
        along_half=view['along_half'],
        offsets=offsets,
        color='tab:purple',
        center_dash_index=center_idx,
        linewidth_main=0.95,
        linewidth_center=0.8,
        alpha_main=0.30,
        alpha_center=0.24,
    )

    if sep.get('type') == 'rectangle':
        c = np.array(sep.get('center', center.tolist()), dtype=float)
        hl = min(float(sep.get('half_length_m', view['along_half'])), view['along_half'] * 1.05)
        hw = min(float(sep.get('half_width_m', lane_w * 0.22)), view['cross_half'] * 1.10)
        rect_local = np.array([[ hw, hl], [-hw, hl], [-hw, -hl], [ hw, -hl]], dtype=float)
        R = np.stack([r, u], axis=1)
        poly = rect_local @ R.T + c
        ax.add_patch(Polygon(poly, fill=False, edgecolor='tab:red', linewidth=1.1, alpha=0.42, zorder=0.1))


def draw_channel(ax: plt.Axes, scene: Dict[str, Any], view: Dict[str, Any]) -> None:
    area = get_area_context(scene)
    center = view['center']
    u = view['u']
    r = view['r']
    width = float(area.get('channel_width_m', 350.0))
    centerline = area.get('channel_centerline', {})

    if centerline.get('type') == 'line_segment':
        a = np.array(centerline.get('start_xy', center.tolist()), dtype=float)
        anchor = project_point_to_line(center, a, u)
    else:
        anchor = center

    draw_local_parallel_lines(
        ax,
        center=anchor,
        u=u,
        r=r,
        along_half=view['along_half'],
        offsets=[-width / 2.0, 0.0, width / 2.0],
        color='tab:green',
        center_dash_index=1,
        linewidth_main=1.0,
        linewidth_center=0.8,
        alpha_main=0.36,
        alpha_center=0.28,
    )


def draw_area_context(ax: plt.Axes, scene: Dict[str, Any], view: Dict[str, Any]) -> None:
    mode = view['mode']
    if mode == 'tss':
        draw_tss(ax, scene, view)
    elif mode == 'channel':
        draw_channel(ax, scene, view)
    elif mode == 'legacy_tss':
        center = view['center']
        u = view['u']
        a = center - u * view['along_half'] * 1.08
        b = center + u * view['along_half'] * 1.08
        ax.plot([a[0], b[0]], [a[1], b[1]], linestyle='--', linewidth=0.8, alpha=0.24, color='tab:purple', zorder=0)


# ============================================================
# Rendering
# ============================================================

def render_topdown(scene: Dict[str, Any], out_path: Path, size_px: int, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    axes = axes.flatten()

    # Snapshot-centered view; reused across offsets for visual comparability.
    snapshot_view = dynamic_view(scene)

    for ax, off in zip(axes, FRAME_OFFSETS_S):
        own, tgts = scene_states_at_offset(scene, off)
        view = dict(snapshot_view)
        center = np.array([float(own['x_m']), float(own['y_m'])], dtype=float)
        # Re-center on the current ownship state, but preserve scale and overlay orientation.
        view['center'] = center

        ax.set_xlim(center[0] - view['x_half'], center[0] + view['x_half'])
        ax.set_ylim(center[1] - view['y_half'], center[1] + view['y_half'])
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linewidth=0.45, alpha=0.28)
        draw_area_context(ax, scene, view)

        entities = [own] + tgts
        for i, ent in enumerate(entities):
            vid = vessel_id(ent)
            disp_len, disp_wid = display_ship_dims(ent, view)
            rect, tri = ship_polygon(float(ent['x_m']), float(ent['y_m']), float(ent['heading_deg']), disp_len, disp_wid)
            z = 5 if i == 0 else 4
            alpha = 0.62 if i == 0 else 0.48
            ax.add_patch(Polygon(rect, closed=True, fill=True, edgecolor='black', linewidth=1.0, alpha=alpha, zorder=z))
            ax.add_patch(Polygon(tri, closed=True, fill=True, edgecolor='black', linewidth=1.0, alpha=min(0.88, alpha + 0.16), zorder=z + 0.1))
            v = heading_to_vec(float(ent['heading_deg'])) * float(ent.get('speed_mps', 0.0)) * 18.0
            ax.arrow(
                float(ent['x_m']), float(ent['y_m']), v[0], v[1],
                head_width=max(math.sqrt((2.0 * view['x_half']) * (2.0 * view['y_half'])) * 0.008, disp_wid * 0.22),
                length_includes_head=True, alpha=0.40, zorder=z + 0.2,
            )
            dx = max(disp_len * 0.45, view['x_half'] * 0.025) * (1.0 if i == 0 or float(ent['x_m']) >= center[0] else -1.0)
            dy = max(disp_len * 0.38, view['y_half'] * 0.022) * (1.0 if i == 0 or float(ent['y_m']) >= center[1] else -1.0)
            ax.text(float(ent['x_m']) + dx, float(ent['y_m']) + dy, vid, fontsize=8, zorder=z + 0.3)

        ax.set_title(f'Δt={int(off):d}s → snapshot')
        ax.text(
            0.98, 0.98,
            f'ENU frame\nrange≈{int(max(view["x_half"], view["y_half"]))}m',
            transform=ax.transAxes,
            ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', alpha=0.30),
        )

    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def render_radar(scene: Dict[str, Any], out_path: Path, size_px: int, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    axes = axes.flatten()
    view = dynamic_view(scene)
    rmax = max(3500.0, max(view['x_half'], view['y_half']) * 1.15)
    rng = np.random.default_rng(0)
    grid_n = 420
    extent = (-rmax, rmax, -rmax, rmax)

    def add_gaussian(img: np.ndarray, x0: float, y0: float, amp: float, sigma_m: float) -> None:
        sx = max(2.0, sigma_m / (2 * rmax) * grid_n)
        ix = int((x0 - extent[0]) / (extent[1] - extent[0]) * (grid_n - 1))
        iy = int((y0 - extent[2]) / (extent[3] - extent[2]) * (grid_n - 1))
        rad = int(max(6, 3 * sx))
        x1, x2 = max(0, ix - rad), min(grid_n - 1, ix + rad)
        y1, y2 = max(0, iy - rad), min(grid_n - 1, iy + rad)
        if x1 >= x2 or y1 >= y2:
            return
        xx = np.arange(x1, x2 + 1) - ix
        yy = np.arange(y1, y2 + 1) - iy
        X, Y = np.meshgrid(xx, yy)
        g = amp * np.exp(-(X * X + Y * Y) / (2 * sx * sx))
        img[y1:y2 + 1, x1:x2 + 1] += g

    for ax, off in zip(axes, FRAME_OFFSETS_S):
        own, tgts = scene_states_at_offset(scene, off)
        img = np.zeros((grid_n, grid_n), dtype=np.float32)
        for _ in range(40):
            cr = rmax * np.sqrt(rng.uniform(0, 1))
            ct = rng.uniform(0, 2 * math.pi)
            add_gaussian(img, cr * math.cos(ct), cr * math.sin(ct), 0.14, 40.0)
        for tgt in tgts:
            rel = np.array([float(tgt['x_m']) - float(own['x_m']), float(tgt['y_m']) - float(own['y_m'])], dtype=float)
            rr = float(np.linalg.norm(rel))
            if rr > rmax:
                continue
            bearing = relative_bearing_deg(float(own['heading_deg']), rel)
            rr_n = max(0.0, rr + float(rng.normal(0, 12.0)))
            bb_n = math.radians(bearing + float(rng.normal(0, 0.9)))
            x = rr_n * math.sin(bb_n)
            y = rr_n * math.cos(bb_n)
            add_gaussian(img, x, y, 1.0, 26.0)
            ax.text(x + 55, y + 55, vessel_id(tgt), fontsize=8)
        vmax = np.percentile(img, 99.5) if img.max() > 0 else 1.0
        ax.imshow(np.clip(img, 0, vmax), extent=extent, origin='lower', alpha=0.95)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-rmax, rmax)
        ax.set_ylim(-rmax, rmax)
        ax.grid(True, linewidth=0.45, alpha=0.28)
        for rr in [rmax * 0.25, rmax * 0.5, rmax * 0.75, rmax]:
            ax.add_patch(plt.Circle((0, 0), rr, fill=False, linewidth=0.8, alpha=0.45))
        ax.set_title(f'Radar Δt={int(off):d}s')
        ax.text(0.98, 0.98, 'Ahead ↑  Starboard →', transform=ax.transAxes,
                ha='right', va='top', fontsize=8, bbox=dict(boxstyle='round,pad=0.2', alpha=0.30))
    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def render_one(row: Dict[str, Any], root: Path, out_dir: Path, size_px: int, dpi: int, overwrite: bool) -> Dict[str, Any]:
    sample_id = row['sample_id']
    family = row.get('pattern') or row.get('family') or 'unknown'
    scene = get_scene_spec(row)
    fam_dir = out_dir / family
    top_path = fam_dir / f'{sample_id}_topdown.png'
    radar_path = fam_dir / f'{sample_id}_radar.png'
    if overwrite or not top_path.exists():
        render_topdown(scene, top_path, size_px=size_px, dpi=dpi)
    if overwrite or not radar_path.exists():
        render_radar(scene, radar_path, size_px=size_px, dpi=dpi)
    return {
        'sample_id': sample_id,
        'family': family,
        'topdown_image': os.path.relpath(top_path, root),
        'radar_image': os.path.relpath(radar_path, root),
    }


# ============================================================
# Gallery
# ============================================================

def build_gallery_pages(root: Path, gallery_dir: Path, index_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    gallery_dir.mkdir(parents=True, exist_ok=True)
    fam2rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in index_rows:
        fam2rows[r['family']].append(r)
    out: Dict[str, str] = {}
    for fam, rows in sorted(fam2rows.items()):
        rows = sorted(rows, key=lambda x: x['sample_id'])
        parts = [
            '<!doctype html><html><head><meta charset="utf-8">',
            f'<title>{html.escape(fam)} gallery</title>',
            '<style>body{font-family:Arial,sans-serif;margin:18px;} .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(460px,1fr));gap:18px;} .card{border:1px solid #ccc;border-radius:10px;padding:10px;} img{max-width:100%;height:auto;border:1px solid #ddd;} .pair{display:grid;grid-template-columns:1fr 1fr;gap:8px;} .meta{font-size:13px;margin:6px 0 8px;color:#333;}</style>',
            '</head><body>',
            f'<h1>{html.escape(fam)} gallery</h1>',
            f'<p>samples: {len(rows)}</p>',
            '<div class="grid">'
        ]
        for r in rows:
            top_rel = os.path.relpath(root / r['topdown_image'], gallery_dir)
            rad_rel = os.path.relpath(root / r['radar_image'], gallery_dir)
            parts += [
                '<div class="card">',
                f'<div class="meta"><strong>{html.escape(r["sample_id"])}</strong></div>',
                '<div class="pair">',
                f'<div><div class="meta">topdown</div><img src="{html.escape(top_rel)}"></div>',
                f'<div><div class="meta">radar</div><img src="{html.escape(rad_rel)}"></div>',
                '</div>',
                '</div>'
            ]
        parts += ['</div></body></html>']
        page = gallery_dir / f'{fam}.html'
        page.write_text('\n'.join(parts), encoding='utf-8')
        out[fam] = os.path.relpath(page, root)
    return out


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    source = root / args.source_jsonl
    out_dir = root / args.out_dir
    gallery_dir = root / args.gallery_dir
    index_jsonl = root / args.index_jsonl
    report_json = root / args.report_json

    rows = read_jsonl(source)
    if args.max_samples and args.max_samples > 0:
        rows = rows[:args.max_samples]

    seen = set()
    uniq_rows: List[Dict[str, Any]] = []
    for row in rows:
        sid = row.get('sample_id')
        if sid in seen:
            continue
        seen.add(sid)
        uniq_rows.append(row)

    index_rows: List[Dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    failures: List[Dict[str, Any]] = []
    for row in uniq_rows:
        try:
            entry = render_one(row, root, out_dir, args.size_px, args.dpi, args.overwrite)
            index_rows.append(entry)
            family_counts[entry['family']] += 1
        except Exception as e:  # noqa: BLE001
            failures.append({'sample_id': row.get('sample_id'), 'family': row.get('pattern'), 'error': repr(e)})

    gallery_pages = build_gallery_pages(root, gallery_dir, index_rows)
    write_jsonl(index_jsonl, index_rows)
    report = {
        'renderer_version': 'native_image_renderer_v3',
        'source_jsonl': os.path.relpath(source, root),
        'rendered_count': len(index_rows),
        'failure_count': len(failures),
        'family_counts': dict(sorted(family_counts.items())),
        'gallery_pages': gallery_pages,
        'index_jsonl': os.path.relpath(index_jsonl, root),
        'out_dir': os.path.relpath(out_dir, root),
        'failures_head': failures[:10],
    }
    write_json(report_json, report)
    print('=' * 100)
    print('render_native_benchmark_images.py')
    print('=' * 100)
    print('Renderer version  : native_image_renderer_v3')
    print('Source JSONL      :', source)
    print('Rendered count    :', len(index_rows))
    print('Failure count     :', len(failures))
    print('Index JSONL       :', index_jsonl)
    print('Gallery dir       :', gallery_dir)
    print('Report JSON       :', report_json)
    print('Family counts     :', dict(sorted(family_counts.items())))


if __name__ == '__main__':
    main()
