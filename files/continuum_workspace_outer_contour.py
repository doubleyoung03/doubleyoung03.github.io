import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# =========================
# Parameters
# =========================
L1 = 25.0
scale = 0.9
N_segments = 18
N_joints = N_segments - 1

deg = np.deg2rad
N_arc = 450

# Angles
J1_MIN, J1_MAX = deg(0), deg(30)            # joint1: [0, +30]
J_CONT_MIN, J_CONT_MAX = deg(-30), deg(30)  # joint2..joint10: [-30, +30]
J_EXT = deg(30)                             # extreme magnitude

# For stages >= joint2
J1_FIXED = deg(30)

# Rotate display so segment (joint1->joint2) points to +y.
# Original direction is +30 deg; target is +90 deg => rotate +60 deg.
ROT_PHI = deg(90) - J1_FIXED

# Build lengths L1..L18
L = [L1]
for _ in range(N_segments - 1):
    L.append(L[-1] * scale)
L = np.array(L)

# tail_sum[i] = sum(L[i:]) for segments i..end (0-based)
tail_sum = np.array([np.sum(L[i:]) for i in range(N_segments)])


# =========================
# Helpers
# =========================
def rotate_xy(x, y, phi):
    x = np.asarray(x)
    y = np.asarray(y)
    c = np.cos(phi)
    s = np.sin(phi)
    xr = c * x - s * y
    yr = s * x + c * y
    return xr, yr


def arc_xy(center, r, ang0, ang1, n=N_arc):
    ang = np.linspace(ang0, ang1, n)
    x = center[0] + r * np.cos(ang)
    y = center[1] + r * np.sin(ang)
    return x, y


def fill_sector(ax, center, x, y, alpha=0.05, lw=1.2, rotate_phi=0.0, polygon_store=None):
    xr, yr = rotate_xy(x, y, rotate_phi)
    cx_arr, cy_arr = rotate_xy(np.array([center[0]]), np.array([center[1]]), rotate_phi)
    cx, cy = float(cx_arr[0]), float(cy_arr[0])

    px = np.concatenate([[cx], xr, [cx]])
    py = np.concatenate([[cy], yr, [cy]])
    ax.fill(px, py, alpha=alpha, linewidth=0)
    ax.plot(xr, yr, linewidth=lw)

    if polygon_store is not None:
        polygon_store.append(np.column_stack((px, py)))


def joint_position_and_base_angle(joint_angles, k_joint):
    """
    joint_angles: list length 17 (joint1..joint17), relative angles
    k_joint: 1..17, return position of joint k (end of segment k)
    base_angle: absolute angle of segment k (sum of joint1..joint(k-1))
    """
    abs_angles = np.zeros(N_segments)
    cum = 0.0
    for s in range(1, N_segments):  # s=1..17 -> seg2..seg18
        cum += joint_angles[s - 1]
        abs_angles[s] = cum

    p = np.array([0.0, 0.0])
    for s in range(0, k_joint):     # segment indices 0..k-1
        p = p + L[s] * np.array([np.cos(abs_angles[s]), np.sin(abs_angles[s])])

    center = p
    base_angle = abs_angles[k_joint - 1]
    return center, base_angle


def joint_range(k_joint, sign10=None):
    """
    Return (a0, a1) for joint k (relative angle range).
    sign10: only used for k>=11 (joint11..joint17)
        sign10 = -1 -> range [0, -30]
        sign10 = +1 -> range [0, +30]
    """
    if k_joint == 1:
        return (J1_MIN, J1_MAX)
    if 2 <= k_joint <= 10:
        return (J_CONT_MIN, J_CONT_MAX)
    if sign10 is None:
        raise ValueError("sign10 must be provided for joint>=11")
    return (0.0, sign10 * J_EXT)


def shift_no_wrap(mask, dy, dx):
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape

    if dy >= 0:
        y_src0, y_src1 = 0, h - dy
        y_dst0, y_dst1 = dy, h
    else:
        y_src0, y_src1 = -dy, h
        y_dst0, y_dst1 = 0, h + dy

    if dx >= 0:
        x_src0, x_src1 = 0, w - dx
        x_dst0, x_dst1 = dx, w
    else:
        x_src0, x_src1 = -dx, w
        x_dst0, x_dst1 = 0, w + dx

    if y_src1 > y_src0 and x_src1 > x_src0:
        out[y_dst0:y_dst1, x_dst0:x_dst1] = mask[y_src0:y_src1, x_src0:x_src1]
    return out


def majority_binary_smooth(mask, rounds=2, threshold=5):
    m = mask.copy()
    for _ in range(max(1, rounds)):
        cnt = np.zeros(m.shape, dtype=np.int16)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cnt += shift_no_wrap(m, dy, dx).astype(np.int16)
        m = cnt >= threshold
    return m


def binary_edge_8n(mask):
    eroded = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            eroded &= shift_no_wrap(mask, dy, dx)
    edge = mask & (~eroded)
    return edge


def contour_perimeter(seg):
    if len(seg) < 2:
        return 0.0
    d = np.diff(seg, axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))


def smooth_closed_contour_fourier(x, y, keep_ratio=0.08):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 16:
        return x, y

    if np.hypot(x[0] - x[-1], y[0] - y[-1]) < 1e-10:
        x = x[:-1]
        y = y[:-1]
    n = len(x)
    if n < 16:
        return np.append(x, x[0]), np.append(y, y[0])

    z = x + 1j * y
    zf = np.fft.fft(z)

    k = int(max(8, min(n // 2, round(n * keep_ratio))))
    filt = np.zeros_like(zf)
    filt[: k + 1] = zf[: k + 1]
    filt[-k:] = zf[-k:]
    z_s = np.fft.ifft(filt)

    x_s = np.real(z_s)
    y_s = np.imag(z_s)
    return np.append(x_s, x_s[0]), np.append(y_s, y_s[0])


def extract_outer_contour_via_edge_pipeline(
    polygons,
    pixels_y=1800,
    padding=6.0,
    bin_threshold=0.5,
    smooth_rounds=2,
    smooth_threshold=5,
    fourier_keep_ratio=0.08,
):
    """
    Pipeline:
      render polygons -> grayscale -> binarize -> edge detect -> boundary -> smooth
    """
    if not polygons:
        raise ValueError("polygons is empty")

    pts = np.vstack(polygons)
    xmin = float(np.min(pts[:, 0]) - padding)
    xmax = float(np.max(pts[:, 0]) + padding)
    ymin = float(np.min(pts[:, 1]) - padding)
    ymax = float(np.max(pts[:, 1]) + padding)

    x_span = max(xmax - xmin, 1e-6)
    y_span = max(ymax - ymin, 1e-6)
    pixels_y = max(900, int(pixels_y))
    pixels_x = int(np.ceil(pixels_y * x_span / y_span))
    pixels_x = max(900, pixels_x)

    dpi = 100
    fig_mask = plt.figure(figsize=(pixels_x / dpi, pixels_y / dpi), dpi=dpi)
    ax_mask = fig_mask.add_axes([0, 0, 1, 1])
    fig_mask.patch.set_facecolor("black")
    ax_mask.set_facecolor("black")
    ax_mask.set_xlim(xmin, xmax)
    ax_mask.set_ylim(ymin, ymax)
    ax_mask.axis("off")

    for poly in polygons:
        ax_mask.fill(poly[:, 0], poly[:, 1], color="white", linewidth=0, antialiased=False)

    fig_mask.canvas.draw()
    rgba = np.asarray(fig_mask.canvas.buffer_rgba())
    plt.close(fig_mask)

    gray = np.mean(rgba[:, :, :3], axis=2) / 255.0
    binary = gray >= float(bin_threshold)
    binary = majority_binary_smooth(binary, rounds=smooth_rounds, threshold=smooth_threshold)

    edge = binary_edge_8n(binary)

    z = np.flipud(edge.astype(float))
    x_grid = np.linspace(xmin, xmax, edge.shape[1])
    y_grid = np.linspace(ymin, ymax, edge.shape[0])

    fig_cont, ax_cont = plt.subplots(figsize=(4, 4))
    cs = ax_cont.contour(x_grid, y_grid, z, levels=[0.5])
    plt.close(fig_cont)

    segs = cs.allsegs[0]
    if len(segs) == 0:
        raise RuntimeError("No boundary detected from edge image")

    best = max(segs, key=contour_perimeter)
    x_raw = best[:, 0]
    y_raw = best[:, 1]

    x_s, y_s = smooth_closed_contour_fourier(x_raw, y_raw, keep_ratio=fourier_keep_ratio)
    return x_s, y_s


# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(10, 10))

joint_centers = {k: [] for k in range(1, N_joints + 1)}
sector_polygons = []

# fixed segment1
base = np.array([0.0, 0.0])
p1 = np.array([L[0], 0.0])
x_seg, y_seg = rotate_xy(np.array([base[0], p1[0]]), np.array([base[1], p1[1]]), ROT_PHI)
ax.plot(x_seg, y_seg, linewidth=3, alpha=0.85, color="tab:blue")
joint_centers[1].append(p1)

# -------------------------
# Stage A: joint1 in [0,30], others = 0
# tip center = joint1 position, radius = sum(L2..L18) = tail_sum[1]
# -------------------------
theta1 = np.linspace(J1_MIN, J1_MAX, N_arc)
r_a = tail_sum[1]
x_a = p1[0] + r_a * np.cos(theta1)
y_a = p1[1] + r_a * np.sin(theta1)
fill_sector(ax, p1, x_a, y_a, alpha=0.08, lw=1.6, rotate_phi=ROT_PHI, polygon_store=sector_polygons)

# -------------------------
# Joint2 stage: joint1=+30, joint2 in [-30,30], others=0
# -------------------------
ja = [0.0] * N_joints
ja[0] = J1_FIXED
p2, base_ang2 = joint_position_and_base_angle(ja, 2)
joint_centers[2].append(p2)

r_2 = tail_sum[2]  # L3..L18
a0, a1 = joint_range(2)
x, y = arc_xy(p2, r_2, base_ang2 + a0, base_ang2 + a1)
fill_sector(ax, p2, x, y, alpha=0.08, lw=1.6, rotate_phi=ROT_PHI, polygon_store=sector_polygons)

# -------------------------
# Joint k = 3..10 stages:
# joint1=+30, joint2..joint(k-1) are extremes ±30, jointk continuous [-30,30], later=0
# -------------------------
alpha_by_k = {3: 0.06, 4: 0.045, 5: 0.035, 6: 0.028, 7: 0.022, 8: 0.018, 9: 0.015, 10: 0.013}
lw_by_k = {3: 1.4, 4: 1.25, 5: 1.15, 6: 1.05, 7: 0.95, 8: 0.9, 9: 0.85, 10: 0.8}

for k in range(3, 11):
    m = k - 2
    for signs in product([-1, +1], repeat=m):
        ja = [0.0] * N_joints
        ja[0] = J1_FIXED
        for idx_s, sgn in enumerate(signs):
            ja[1 + idx_s] = sgn * J_EXT

        center, base_ang = joint_position_and_base_angle(ja, k)
        joint_centers[k].append(center)

        r_k = tail_sum[k]
        a0, a1 = joint_range(k)
        x, y = arc_xy(center, r_k, base_ang + a0, base_ang + a1)
        fill_sector(
            ax,
            center,
            x,
            y,
            alpha=alpha_by_k.get(k, 0.01),
            lw=lw_by_k.get(k, 0.8),
            rotate_phi=ROT_PHI,
            polygon_store=sector_polygons,
        )

# -------------------------
# Joint k = 11..17 stages:
# joint10 fixed to ±30; this decides one-sided ranges for joint11..17
# -------------------------
alpha_late = 0.010
lw_late = 0.55
signs_2_to_9_all = list(product([-1, +1], repeat=8))  # 256 combos

for sign10 in (-1, +1):
    for k in range(11, N_joints + 1):
        for signs_2_to_9 in signs_2_to_9_all:
            ja = [0.0] * N_joints
            ja[0] = J1_FIXED

            for idx_s, sgn in enumerate(signs_2_to_9):
                ja[1 + idx_s] = sgn * J_EXT  # joint2..joint9

            ja[9] = sign10 * J_EXT  # joint10

            for j in range(11, k):  # joints 11..k-1 already at same sign
                ja[j - 1] = sign10 * J_EXT

            center, base_ang = joint_position_and_base_angle(ja, k)
            joint_centers[k].append(center)

            r_k = tail_sum[k]
            a0, a1 = joint_range(k, sign10=sign10)
            x, y = arc_xy(center, r_k, base_ang + a0, base_ang + a1)
            fill_sector(
                ax,
                center,
                x,
                y,
                alpha=alpha_late,
                lw=lw_late,
                rotate_phi=ROT_PHI,
                polygon_store=sector_polygons,
            )

# -------------------------
# Outer contour via: grayscale -> binary -> edge detect -> smooth boundary
# -------------------------
x_contour, y_contour = extract_outer_contour_via_edge_pipeline(
    sector_polygons,
    pixels_y=1900,
    padding=4.0,
    bin_threshold=0.5,
    smooth_rounds=2,
    smooth_threshold=5,
    fourier_keep_ratio=0.07,
)
ax.plot(x_contour, y_contour, color="crimson", linewidth=3.0, zorder=9, label="Outer contour")

# Optionally show joint center points (rotated for display)
show_joint_points = True
if show_joint_points:
    xs, ys = [], []
    for k in range(1, N_joints + 1):
        for p in joint_centers[k]:
            xr, yr = rotate_xy(np.array([p[0]]), np.array([p[1]]), ROT_PHI)
            xs.append(float(xr[0]))
            ys.append(float(yr[0]))
    ax.scatter(xs, ys, s=6, alpha=0.35, zorder=6, color="black")

# Formatting
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_title("Workspace rotated to +y with edge-detected smooth boundary")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper right")
plt.show()

# quick counts (sanity)
for k in [1, 2, 3, 4, 5, 10, 11, 12, 17]:
    print(f"joint{k} centers stored:", len(joint_centers[k]))
