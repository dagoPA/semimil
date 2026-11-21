#!/usr/bin/env python
"""Method figure: the proposed detector on one test image, with real intermediate data.

Top row, four numbered steps on the test image (small defect circled): (1) the frozen
DINOv2 encodes it into two token representations, shown as PCA color maps (r1: block 8;
r2: block 10 with 3 x 3 average pooling); (2) each token is matched to its nearest token in
the memory bank (real token crops, a rare normal structure outlined); (3) the distances form
one map per representation, and a red path follows the defect token to its nearest
memory-bank token and to its cell of the block-8 map; (4) the standardized scores are
averaged and compared with the threshold at specificity 0.99. Bottom row: the memory bank
is built from defect-free training images, and a schematic of why coverage matters (a
random subsample leaves rare normal tokens uncovered, a greedy coreset keeps them).

The token PCA needs one DINOv2 forward pass and is cached; everything else reads the
outputs of scripts/run_bigbank.py. Argument-free: reads configs/method_overview.yaml.

    uv run python scripts/plot_method_overview.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "method_overview.yaml"

INK, MUTED, PANEL, NAVY = "#1d2330", "#5a6474", "#f3f5f9", "#1f3a5f"
BLUE, ORANGE, RED, GREEN, ARROW = "#2b6cb0", "#dd6b20", "#d1342c", "#0f8a6b", "#475569"
EXTS = {".jpg", ".jpeg", ".png"}


def images(d: Path):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS)


def token_pca(cfg, example: Path, train: list[Path]) -> dict:
    """PCA colour maps (37 x 37 x 3) of the two representations; cached after the first run."""
    cache = Path(cfg["cache"])
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    import torch

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from run_patchcore import build_extractor, load_image

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    paths = [example] + train[: cfg["pca_normals"]]
    out = {}
    for key, layers, pool in (("r1", [7], 0), ("r2", [9], 3)):
        ext = build_extractor({"type": "dinov2", "dino_name": cfg["dino_name"], "layers": layers,
                               "pool": pool}, dev)
        with torch.no_grad():
            f = ext(torch.stack([load_image(p, cfg["input_size"]) for p in paths])).float().cpu().numpy()
        flat = f.reshape(-1, f.shape[-1])
        mu = flat.mean(0)
        _, _, vt = np.linalg.svd(flat - mu, full_matrices=False)
        proj = (f[0].reshape(-1, f.shape[-1]) - mu) @ vt[:3].T
        lo, hi = np.percentile(proj, 1, axis=0), np.percentile(proj, 99, axis=0)
        out[key] = np.clip((proj - lo) / (hi - lo), 0, 1).reshape(f.shape[1], f.shape[2], 3)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def image_scores(npz: str):
    z = np.load(npz)
    t = z["test_map"].astype(np.float32)
    c = z["calib_map"].astype(np.float32)
    st, sc = t.reshape(len(t), -1).max(1), c.reshape(len(c), -1).max(1)
    return t, (st - sc.mean()) / sc.std()


def token_crop(path: Path, r: int, c: int, size: int, ctx: int = 2) -> np.ndarray:
    a = np.asarray(Image.open(path).convert("L").resize((size, size), Image.BILINEAR))
    k = ctx * 14
    a = np.pad(a, k, constant_values=255)
    y, x = r * 14 + k, c * 14 + k
    return a[y - k:y + 14 + k, x - k:x + 14 + k]


def heat_overlay(gray: np.ndarray, dmap: np.ndarray) -> np.ndarray:
    """Distance map in a thermal colour map over the image; low distances stay transparent."""
    lo, hi = np.percentile(dmap, 50), dmap.max()
    h = np.clip((dmap - lo) / (hi - lo), 0, 1)
    h = np.asarray(Image.fromarray((h * 255).astype(np.uint8)).resize(gray.shape[::-1], Image.BILINEAR),
                   dtype=np.float32) / 255
    rgb = matplotlib.colormaps["inferno"](0.25 + 0.75 * h)[..., :3]
    alpha = (0.15 + 0.8 * h ** 0.9)[..., None]
    g = 0.85 * np.repeat(gray[..., None].astype(np.float32) / 255, 3, axis=2)
    return np.clip(g * (1 - alpha) + rgb * alpha, 0, 1)


def soften(rgb: np.ndarray) -> np.ndarray:
    """PCA colours with less saturation, so that they sit with the rest of the figure."""
    hsv = matplotlib.colors.rgb_to_hsv(rgb)
    hsv[..., 1] *= 0.7
    hsv[..., 2] = 0.2 + 0.78 * hsv[..., 2]
    return matplotlib.colors.hsv_to_rgb(hsv)


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open())
    raw = Path(cfg["data"]["raw_dir"])
    d = cfg["data"]
    norm, abn = images(raw / d["test_normal_dir"]), images(raw / d["test_abnormal_dir"])
    train = sorted(raw.glob(d["train_normal_glob"]))
    ex = next(p for p in abn if p.stem == cfg["example"])
    idx = len(norm) + abn.index(ex)
    mask = np.asarray(Image.open(raw / d["test_mask_dir"] / f"{ex.stem}_mask.jpg").convert("L")) > 127

    pca = token_pca(cfg, ex, train)
    t1, z1 = image_scores(cfg["maps"]["r1"])
    t2, z2 = image_scores(cfg["maps"]["r2"])
    z = (z1 + z2) / 2
    tau = np.percentile(z[:len(norm)], 99)
    gray = np.asarray(Image.open(ex).convert("L").resize((400, 400), Image.BILINEAR))
    g = t1.shape[1]
    tok_mask = mask.reshape(g, mask.shape[0] // g, g, -1).any((1, 3)) if mask.shape[0] % g == 0 else \
        np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((g, g), Image.BOX)) > 0
    er, ec = cfg["example_token"]
    top = np.unravel_index(np.argmax(np.where(tok_mask, t1[idx], -np.inf)), (g, g))
    if tuple(int(v) for v in top) != (er, ec):
        sys.exit(f"highest-scoring defect token is {tuple(top)}, config says {(er, ec)}")
    dval = float(t1[idx][er, ec])

    plt.rcParams.update({"font.family": "Nimbus Sans", "font.size": 7, "mathtext.fontset": "custom",
                         "mathtext.rm": "Nimbus Sans", "mathtext.it": "Nimbus Sans:italic",
                         "mathtext.bf": "Nimbus Sans:bold"})
    W, H = 7.0, 3.3
    fig = plt.figure(figsize=(W, H))
    ov = fig.add_axes((0, 0, 1, 1), zorder=10)
    ov.set_xlim(0, W), ov.set_ylim(0, H), ov.axis("off")
    ov.patch.set_alpha(0)
    bg = fig.add_axes((0, 0, 1, 1), zorder=-1)
    bg.set_xlim(0, W), bg.set_ylim(0, H), bg.axis("off")

    def panel(x0, x1, y0, y1, num, text):
        bg.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.07",
                                    fc=PANEL, ec="none"))
        if num:
            bg.add_patch(Circle((x0 + 0.13, y1 - 0.13), 0.075, color=NAVY))
            bg.text(x0 + 0.13, y1 - 0.13, num, color="white", ha="center", va="center", fontsize=6.5,
                    fontweight="bold")
        bg.text(x0 + (0.25 if num else 0.08), y1 - 0.13, text, ha="left", va="center", fontsize=7.6,
                fontweight="bold", color=INK)

    def picture(x, y, w, h, arr, frame=None, lw=1.3, **kw):
        ax = fig.add_axes((x / W, y / H, w / W, h / H))
        ax.set_xticks([]), ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        im = ax.imshow(arr, **kw)
        clip = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.07", transform=ax.transAxes)
        im.set_clip_path(clip)
        if frame:
            ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.07",
                                        transform=ax.transAxes, fill=False, ec=frame, lw=lw, clip_on=False))
        return ax

    def arrow(p, q, color=ARROW, lw=1.1, cs="arc3", ls="-", style="-|>", ms=8, z=3):
        ov.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls,
                                     connectionstyle=cs, shrinkA=0, shrinkB=0, zorder=z,
                                     capstyle="round", joinstyle="round"))

    def note(x, y, text, color=MUTED, size=6.2, ha="center", va="top", **kw):
        ov.text(x, y, text, color=color, fontsize=size, ha=ha, va=va, linespacing=1.15, **kw)

    Y0, Y1 = 0.95, 3.25
    panel(0.03, 0.98, Y0, Y1, "", "Test image")
    panel(1.04, 2.74, Y0, Y1, "1", "Encode with frozen DINOv2")
    panel(2.80, 4.26, Y0, Y1, "2", "Memory-bank search")
    panel(4.32, 5.33, Y0, Y1, "3", "Distance maps")
    panel(5.39, 6.97, Y0, Y1, "4", "Image score")

    # ---- test image
    picture(0.08, 1.62, 0.85, 0.85, gray, cmap="gray", vmin=0, vmax=255)
    ys, xs = np.nonzero(mask)
    k = 0.85 / mask.shape[0]
    ov.add_patch(Circle((0.08 + xs.mean() * k, 1.62 + 0.85 - ys.mean() * k), 0.055, fill=False, ec=RED, lw=1.2))
    note(0.505, 1.56, "SEM image, 512" + "\u00d7" + "512 px,\nsmall defect circled", size=5.8)

    # ---- 1. DINOv2 module and token maps
    mx0, mx1, my0, my1 = 1.13, 1.69, 1.18, 2.95
    ov.add_patch(FancyBboxPatch((mx0, my0), mx1 - mx0, my1 - my0, boxstyle="round,pad=0,rounding_size=0.06",
                                fc="#dbe7f6", ec="#9db8d9", lw=0.8, zorder=1))
    ov.text((mx0 + mx1) / 2, my1 - 0.1, "DINOv2", ha="center", va="top", fontsize=7.4, fontweight="bold",
            color=NAVY)
    ov.text((mx0 + mx1) / 2, my1 - 0.25, "ViT-S/14", ha="center", va="top", fontsize=6.2, color=NAVY)
    sx0, sy0, sh, sdx = 1.2, 1.47, 0.95, 0.062
    for i in range(6):
        x = sx0 + i * sdx
        ov.add_patch(Polygon([(x, sy0), (x + 0.03, sy0 + 0.07), (x + 0.03, sy0 + sh + 0.07), (x, sy0 + sh)],
                             closed=True, fc="#ffffff", ec="#7f9cc0", lw=0.5, zorder=2))
    ov.text((mx0 + mx1) / 2, my0 + 0.07, "12 blocks,\nfrozen", ha="center", va="bottom", fontsize=5.8,
            color=NAVY, linespacing=1.1)
    arrow((0.95, 2.05), (1.12, 2.05))
    tap1, tap2 = (sx0 + 3 * sdx + 0.03, sy0 + 0.8), (sx0 + 4 * sdx + 0.03, sy0 + 0.3)
    ov.add_patch(Circle(tap1, 0.024, color=BLUE, zorder=4))
    ov.add_patch(Circle(tap2, 0.024, color=ORANGE, zorder=4))
    m = 0.64
    r1x, r1y, r2y = 1.97, 2.22, 1.3
    picture(r1x, r1y, m, m, soften(pca["r1"]), frame=BLUE, interpolation="nearest")
    picture(r1x, r2y, m, m, soften(pca["r2"]), frame=ORANGE, interpolation="nearest")
    arrow(tap1, (r1x - 0.02, r1y + m / 2), color=BLUE, cs="arc3,rad=-0.2")
    arrow(tap2, (r1x - 0.02, r2y + m / 2), color=ORANGE, cs="arc3,rad=0.2")
    note(r1x + m / 2, r1y - 0.03, r"block 8 tokens, $r_1$", color=BLUE)
    note(r1x + m / 2, r2y - 0.03, "block 10 tokens, 3" + "\u00d7" + "3\naverage pooling, " + r"$r_2$", color=ORANGE)

    # ---- 2. memory of normal tokens
    t, gp = 0.37, 0.04
    gx, gy = 2.80 + (1.46 - 3 * t - 2 * gp) / 2, 1.5
    near_xy = None
    for n, (stem, r, c, role) in enumerate(cfg["memory_tokens"]):
        tx, ty = gx + (n % 3) * (t + gp), gy + (2 - n // 3) * (t + gp)
        frame = {"nearest": RED, "rare": GREEN}.get(role)
        picture(tx, ty, t, t, token_crop(raw / "Anomaly_train" / f"{stem}.jpg", r, c, cfg["input_size"]),
                frame=frame, lw=1.6, cmap="gray", vmin=0, vmax=255)
        if role == "nearest":
            near_xy = (tx + t, ty + t / 2)
    note(3.53, 1.43, "every token of 23,688 normal images\n(32.4 M), or a greedy 1 % coreset", color=INK,
         size=6)
    note(3.53, 1.14, "green: a rare normal token", color=GREEN, size=5.8)
    for y in (r1y + m / 2, r2y + m / 2):
        arrow((r1x + m + 0.02, y), (gx - 0.03, y), color=BLUE if y > 2 else ORANGE)

    # ---- 3. distance maps, and how one cell is computed
    dx_ = 4.5
    picture(dx_, r1y, m, m, heat_overlay(gray, t1[idx]), frame=BLUE)
    picture(dx_, r2y, m, m, heat_overlay(gray, t2[idx]), frame=ORANGE)
    for y in (r1y + m / 2, r2y + m / 2):
        arrow((gx + 3 * t + 2 * gp + 0.03, y), (dx_ - 0.03, y), color=BLUE if y > 2 else ORANGE)
    note(dx_ + m / 2, r1y - 0.03, r"$d_1$, also localizes", color=BLUE)
    note(dx_ + m / 2, r2y - 0.03, r"$d_2$", color=ORANGE)
    note(4.825, 3.0, r"$d(p)=\min_m \Vert f(p)-m\Vert$", color=INK, size=6.4)
    cell = m / g
    c_r1 = (r1x + (ec + 0.5) * cell, r1y + m - (er + 0.5) * cell)
    c_d1 = (dx_ + (ec + 0.5) * cell, r1y + m - (er + 0.5) * cell)
    for (cx, cy) in (c_r1, c_d1):
        ov.add_patch(FancyBboxPatch((cx - cell * 1.3, cy - cell * 1.3), cell * 2.6, cell * 2.6,
                                    boxstyle="round,pad=0,rounding_size=0.01", fill=False, ec=RED, lw=1.1,
                                    zorder=6))
    arrow((c_r1[0], c_r1[1] + cell * 1.3), (gx + t / 2, gy + 3 * t + 2 * gp + 0.01), color=RED, lw=0.9,
          ls=(0, (3, 2)), cs="arc3,rad=-0.35", ms=6, z=6)
    arrow((gx + t / 2 + 0.05, gy + 3 * t + 2 * gp + 0.01), (c_d1[0], c_d1[1] + cell * 1.3), color=RED,
          lw=0.9, ls=(0, (3, 2)), cs="arc3,rad=-0.3", ms=6, z=6)
    ov.text(3.93, 2.79, f"d = {dval:.1f}", color=RED, fontsize=6.2, ha="center", va="center", zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", fc=PANEL, ec="none"))
    note(4.825, 1.12, "each cell: distance of one\npatch to the memory bank", size=5.6)

    # ---- 4. image score
    node = (5.5, r1y - 0.12)
    ov.add_patch(Circle(node, 0.075, fc="#ffffff", ec=INK, lw=0.9, zorder=4))
    ov.text(*node, "+", ha="center", va="center", fontsize=8, zorder=5)
    arrow((dx_ + m + 0.03, r1y + m / 2), (node[0] - 0.035, node[1] + 0.07), color=BLUE, cs="arc3,rad=-0.25")
    arrow((dx_ + m + 0.03, r2y + m / 2), (node[0] - 0.035, node[1] - 0.07), color=ORANGE, cs="arc3,rad=0.25")
    arrow((node[0] + 0.08, node[1]), (5.74, node[1]))
    hx = fig.add_axes((5.78 / W, 1.5 / H, 0.98 / W, 1.2 / H))
    hx.set_facecolor("none")
    hx.hist(z[:len(norm)], bins=np.linspace(-2.5, 4.5, 43), color="#b9c6d6", edgecolor="#ffffff", linewidth=0.3)
    hx.axvline(tau, color=NAVY, lw=0.9, ls=(0, (3, 2)))
    ymax = hx.get_ylim()[1]
    hx.annotate("", xy=(z[idx], 0), xytext=(z[idx], ymax * 0.55),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.1, mutation_scale=7))
    hx.text(tau + 0.2, ymax * 0.6, "test image:\n" + rf"$z$ = {z[idx]:.2f} > $\tau$", color=RED, ha="left",
            va="bottom", fontsize=5.8,
            linespacing=1.05)
    hx.text(tau + 0.2, ymax * 0.98, r"$\tau$: specificity" + "\n0.99", ha="left", va="top", fontsize=5.8,
            linespacing=1.05, color=NAVY)
    hx.text(-0.4, ymax * 1.03, "normal test images", color=MUTED, ha="center", va="bottom", fontsize=5.8)
    hx.set_xlim(-2.6, 4.6), hx.set_yticks([]), hx.set_xticks([-2, 0, 2, 4])
    hx.set_xlabel(r"$z=(z_1+z_2)/2$", fontsize=6.5, labelpad=1, color=INK)
    note(6.18, 1.17, "each score standardized on 200 normal\ncalibration images, then averaged", size=5.6)
    hx.tick_params(labelsize=6, length=2, pad=1, colors=MUTED)
    for s_ in ("top", "right", "left"):
        hx.spines[s_].set_visible(False)
    hx.spines["bottom"].set_color(MUTED), hx.spines["bottom"].set_linewidth(0.6)

    # ---- bottom: building the memory, and why coverage matters
    panel(0.03, 4.26, 0.04, 0.88, "", "")
    bg.text(0.11, 0.76, "Building the memory bank once, from defect-free images (no labels)", ha="left",
            va="center", fontsize=6.6, fontweight="bold", color=INK)
    for j, stem in enumerate(cfg["setup_images"]):
        picture(0.13 + 0.08 * j, 0.1 + 0.06 * (2 - j), 0.44, 0.44,
                np.asarray(Image.open(raw / "Anomaly_train" / f"{stem}.jpg").convert("L")), frame="#ffffff",
                lw=0.8, cmap="gray", vmin=0, vmax=255)
    note(0.78, 0.37, "23,688\ntraining\nimages", color=INK, size=6, ha="left", va="center")
    ov.add_patch(FancyBboxPatch((1.44, 0.2), 0.5, 0.34, boxstyle="round,pad=0,rounding_size=0.05",
                                fc="#dbe7f6", ec="#9db8d9", lw=0.8, zorder=2))
    ov.text(1.69, 0.37, "DINOv2", ha="center", va="center", fontsize=6.6, fontweight="bold", color=NAVY, zorder=3)
    arrow((1.2, 0.37), (1.42, 0.37))
    arrow((1.96, 0.37), (3.53, 0.37), cs="arc3")
    arrow((3.53, 0.37), (3.53, 0.98), cs="arc3")
    note(2.75, 0.43, "every token, or a greedy 1 % coreset", size=6, va="bottom")

    panel(4.32, 6.97, 0.04, 0.88, "", "")
    bg.text(4.4, 0.76, "Why coverage matters (illustration)", ha="left", va="center", fontsize=6.6,
            fontweight="bold", color=INK)
    rng = np.random.default_rng(1)
    common = rng.normal(0, 1, (400, 2)) * [1.0, 0.7]
    rare = np.array([[3.4, 2.0], [3.9, -1.2], [-3.6, 1.7], [-3.0, -2.2], [0.4, 3.1], [4.4, 0.6]])
    pts = np.vstack([common, rare])
    is_rare = np.r_[np.zeros(len(common), bool), np.ones(len(rare), bool)]
    rand = rng.choice(len(pts), 30, replace=False)
    sel = [int(np.argmin(np.linalg.norm(pts, axis=1)))]
    for _ in range(29):  # farthest-point (greedy coreset) selection
        dd = np.linalg.norm(pts[:, None] - pts[sel][None], axis=2).min(1)
        sel.append(int(np.argmax(dd)))
    for j, (name, chosen, col) in enumerate((("random subsample", rand, "#334155"), ("greedy coreset", sel, GREEN))):
        a = fig.add_axes(((4.42 + 1.3 * j) / W, 0.07 / H, 1.15 / W, 0.4 / H))
        a.set_facecolor("#ffffff")
        a.scatter(pts[:, 0], pts[:, 1], s=1.2, color="#cfd6de", lw=0)
        a.scatter(pts[chosen, 0], pts[chosen, 1], s=5, color=col, lw=0)
        missed = [i for i in np.flatnonzero(is_rare) if i not in set(chosen)]
        a.scatter(pts[missed, 0], pts[missed, 1], s=16, facecolors="none", edgecolors=RED, lw=0.7)
        a.set_xticks([]), a.set_yticks([])
        a.set_xlim(-5, 5.4), a.set_ylim(-3, 3.8)
        for s_ in a.spines.values():
            s_.set_visible(False)
        a.text(0.0, 1.03, name, transform=a.transAxes, fontsize=6, color=col, ha="left", va="bottom")
    note(4.4, 0.64, "red rings: rare normal tokens left uncovered", color=RED, size=5.8, ha="left",
         va="center")

    out = Path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(Path(cfg["cache"]).with_name("method_overview_preview.png"), dpi=300)
    print(f"wrote {out} (z = {z[idx]:.2f}, tau = {tau:.2f}, defect token d = {dval:.2f})")


if __name__ == "__main__":
    main()
