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


def fill_sector(ax, center, x, y, alpha=0.05, lw=1.2, rotate_phi=0.0, boundary_store=None):
    xr, yr = rotate_xy(x, y, rotate_phi)
    cx_arr, cy_arr = rotate_xy(np.array([center[0]]), np.array([center[1]]), rotate_phi)
    cx, cy = float(cx_arr[0]), float(cy_arr[0])

    px = np.concatenate([[cx], xr, [cx]])
    py = np.concatenate([[cy], yr, [cy]])
    ax.fill(px, py, alpha=alpha)
    ax.plot(xr, yr, linewidth=lw)

    # Store arc + two radial edges for contour extraction.
    if boundary_store is not None:
        boundary_store.append(np.column_stack((xr, yr)))
        n_rad = max(40, len(x) // 6)
        t = np.linspace(0.0, 1.0, n_rad)
        boundary_store.append(np.column_stack((cx + t * (xr[0] - cx), cy + t * (yr[0] - cy))))
        boundary_store.append(np.column_stack((cx + t * (xr[-1] - cx), cy + t * (yr[-1] - cy))))


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


def moving_average(x, window):
    if window <= 1:
        return x.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x_pad, kernel, mode="valid")


def symmetric_outer_contour(boundary_segments, n_bins=900, smooth_window=35):
    """
    Extract one smooth half-contour x>=0 as x_half(y), then mirror to x<=0.
    This enforces y-axis symmetry in plotting.
    """
    if not boundary_segments:
        raise ValueError("boundary_segments is empty")

    pts = np.vstack(boundary_segments)
    x = pts[:, 0]
    y = pts[:, 1]

    y_min, y_max = float(np.min(y)), float(np.max(y))
    bins = np.linspace(y_min, y_max, n_bins + 1)
    y_mid = 0.5 * (bins[:-1] + bins[1:])
    idx = np.digitize(y, bins) - 1

    x_half = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = idx == i
        if np.any(mask):
            x_half[i] = np.max(np.abs(x[mask]))

    valid = ~np.isnan(x_half)
    if np.count_nonzero(valid) < 20:
        raise RuntimeError("Not enough points to build contour")

    y_valid = y_mid[valid]
    x_valid = x_half[valid]

    y_dense = np.linspace(float(np.min(y_valid)), float(np.max(y_valid)), 1400)
    x_dense = np.interp(y_dense, y_valid, x_valid)
    x_smooth = moving_average(x_dense, smooth_window)

    # Make top/bottom connection smooth when left/right curves meet.
    cap = max(16, len(x_smooth) // 70)
    x_smooth[:cap] *= np.linspace(0.0, 1.0, cap)
    x_smooth[-cap:] *= np.linspace(1.0, 0.0, cap)
    return x_smooth, y_dense


# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(10, 10))

joint_centers = {k: [] for k in range(1, N_joints + 1)}
boundary_segments = []

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
fill_sector(ax, p1, x_a, y_a, alpha=0.08, lw=1.6, rotate_phi=ROT_PHI, boundary_store=boundary_segments)

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
fill_sector(ax, p2, x, y, alpha=0.08, lw=1.6, rotate_phi=ROT_PHI, boundary_store=boundary_segments)

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
            boundary_store=boundary_segments,
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
                boundary_store=boundary_segments,
            )

# -------------------------
# Outer contour: draw right/left halves (smoothly connected at ends)
# -------------------------
x_half, y_half = symmetric_outer_contour(boundary_segments, n_bins=950, smooth_window=35)
ax.plot(x_half, y_half, color="crimson", linewidth=2.8, zorder=8, label="Outer contour (right)")
ax.plot(-x_half, y_half, color="crimson", linewidth=2.8, zorder=8, label="Outer contour (left)")

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
ax.set_title("Workspace rotated to +y with smooth symmetric outer contour")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper right")
plt.show()

# quick counts (sanity)
for k in [1, 2, 3, 4, 5, 10, 11, 12, 17]:
    print(f"joint{k} centers stored:", len(joint_centers[k]))
