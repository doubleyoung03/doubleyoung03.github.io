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


def circular_moving_average(arr, window):
    if window <= 1:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    ext = np.concatenate((arr[-pad:], arr, arr[:pad]))
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(ext, kernel, mode="valid")


def polygon_area(x, y):
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


def smooth_closed_contour(x, y, window=29, passes=2):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 8:
        return x, y

    if np.hypot(x[0] - x[-1], y[0] - y[-1]) < 1e-10:
        x = x[:-1]
        y = y[:-1]

    n = len(x)
    if n < 8:
        return np.append(x, x[0]), np.append(y, y[0])

    max_window = n - 1 if (n - 1) % 2 == 1 else n - 2
    if max_window < 3:
        max_window = 3
    window = min(window, max_window)
    if window % 2 == 0:
        window -= 1
    window = max(window, 3)

    for _ in range(max(1, passes)):
        x = circular_moving_average(x, window)
        y = circular_moving_average(y, window)

    return np.append(x, x[0]), np.append(y, y[0])


def enforce_y_axis_symmetry(mask, xmin, xmax):
    sym = mask.copy()
    h, w = sym.shape
    if xmax == xmin or w < 2:
        return sym

    col0 = int(round((0.0 - xmin) / (xmax - xmin) * (w - 1)))
    col0 = int(np.clip(col0, 0, w - 1))
    max_d = min(col0, w - 1 - col0)

    for d in range(max_d + 1):
        c_l = col0 - d
        c_r = col0 + d
        merged = sym[:, c_l] | sym[:, c_r]
        sym[:, c_l] = merged
        sym[:, c_r] = merged
    return sym


def extract_outer_contour_from_polygons(
    polygons,
    pixels_y=1500,
    padding=8.0,
    enforce_symmetry=True,
    smooth_window=29,
    smooth_passes=2,
):
    if not polygons:
        raise ValueError("polygons is empty")

    pts = np.vstack(polygons)
    xmin = float(np.min(pts[:, 0]) - padding)
    xmax = float(np.max(pts[:, 0]) + padding)
    ymin = float(np.min(pts[:, 1]) - padding)
    ymax = float(np.max(pts[:, 1]) + padding)

    x_span = max(xmax - xmin, 1e-6)
    y_span = max(ymax - ymin, 1e-6)
    pixels_x = int(np.ceil(pixels_y * x_span / y_span))
    pixels_x = max(900, pixels_x)
    pixels_y = max(900, int(pixels_y))

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

    mask = rgba[:, :, 0] > 16
    if enforce_symmetry:
        mask = enforce_y_axis_symmetry(mask, xmin, xmax)

    z = np.flipud(mask.astype(float))
    x_grid = np.linspace(xmin, xmax, mask.shape[1])
    y_grid = np.linspace(ymin, ymax, mask.shape[0])

    fig_cont, ax_cont = plt.subplots(figsize=(4, 4))
    cs = ax_cont.contour(x_grid, y_grid, z, levels=[0.5])
    plt.close(fig_cont)

    segs = cs.allsegs[0]
    if len(segs) == 0:
        raise RuntimeError("No contour found from workspace mask")

    best = max(segs, key=lambda seg: abs(polygon_area(seg[:, 0], seg[:, 1])))
    x_raw = best[:, 0]
    y_raw = best[:, 1]
    x_s, y_s = smooth_closed_contour(x_raw, y_raw, window=smooth_window, passes=smooth_passes)
    return x_s, y_s


def split_contour_right_left(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 12:
        return x, y, x, y

    if np.hypot(x[0] - x[-1], y[0] - y[-1]) < 1e-10:
        x = x[:-1]
        y = y[:-1]
    n = len(x)

    top_thr = np.percentile(y, 85)
    bot_thr = np.percentile(y, 15)
    top_candidates = np.where(y >= top_thr)[0]
    bot_candidates = np.where(y <= bot_thr)[0]
    if len(top_candidates) == 0 or len(bot_candidates) == 0:
        return np.append(x, x[0]), np.append(y, y[0]), np.append(x, x[0]), np.append(y, y[0])

    i_top = int(top_candidates[np.argmin(np.abs(x[top_candidates]))])
    i_bot = int(bot_candidates[np.argmin(np.abs(x[bot_candidates]))])
    if i_top == i_bot:
        i_bot = (i_top + n // 2) % n

    order = np.concatenate((np.arange(i_top, n), np.arange(0, i_top)))
    xo = x[order]
    yo = y[order]
    i_bot_o = int(np.where(order == i_bot)[0][0])

    xa = xo[: i_bot_o + 1]
    ya = yo[: i_bot_o + 1]
    xb = np.concatenate((xo[i_bot_o:], xo[:1]))
    yb = np.concatenate((yo[i_bot_o:], yo[:1]))

    if np.mean(xa) >= np.mean(xb):
        xr, yr = xa, ya
        xl, yl = xb, yb
    else:
        xr, yr = xb, yb
        xl, yl = xa, ya

    if yr[0] < yr[-1]:
        xr = xr[::-1]
        yr = yr[::-1]
    if yl[0] < yl[-1]:
        xl = xl[::-1]
        yl = yl[::-1]
    return xr, yr, xl, yl


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
# Outer contour from binary union mask (closed + smooth)
# -------------------------
x_contour, y_contour = extract_outer_contour_from_polygons(
    sector_polygons,
    pixels_y=1500,
    padding=10.0,
    enforce_symmetry=True,
    smooth_window=29,
    smooth_passes=2,
)

try:
    x_r, y_r, x_l, y_l = split_contour_right_left(x_contour, y_contour)
    ax.plot(x_r, y_r, color="crimson", linewidth=2.9, zorder=9, label="Outer contour (right)")
    ax.plot(x_l, y_l, color="crimson", linewidth=2.9, zorder=9, label="Outer contour (left)")
except Exception:
    ax.plot(x_contour, y_contour, color="crimson", linewidth=2.9, zorder=9, label="Outer contour")

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
ax.set_title("Workspace rotated to +y with smooth closed outer contour")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper right")
plt.show()

# quick counts (sanity)
for k in [1, 2, 3, 4, 5, 10, 11, 12, 17]:
    print(f"joint{k} centers stored:", len(joint_centers[k]))
