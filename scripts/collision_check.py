import numpy as np

BLOCK_L = 0.050   # 5 cm
BLOCK_W = 0.021   # 2.1 cm
BLOCK_H = 0.013   # 1.3 cm


def quat_wxyz_to_yaw(q):
    w, x, y, z = q
    # yaw around z
    return np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z)
    )


def rectangle_corners_xy(center, yaw, length=BLOCK_L, width=BLOCK_W):
    x, y = center[0], center[1]

    local = np.array([
        [ length / 2,  width / 2],
        [ length / 2, -width / 2],
        [-length / 2, -width / 2],
        [-length / 2,  width / 2],
    ])

    R = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)],
    ])

    return local @ R.T + np.array([x, y])


def project_polygon(axis, points):
    dots = points @ axis
    return np.min(dots), np.max(dots)


def polygons_overlap(poly_a, poly_b, margin=0.003):
    axes = []

    for poly in [poly_a, poly_b]:
        for i in range(len(poly)):
            edge = poly[(i + 1) % len(poly)] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)

            if norm > 1e-9:
                axes.append(normal / norm)

    for axis in axes:
        min_a, max_a = project_polygon(axis, poly_a)
        min_b, max_b = project_polygon(axis, poly_b)

        # margin makes the check stricter
        if max_a + margin < min_b or max_b + margin < min_a:
            return False

    return True

def same_layer(z1, z2, tolerance=0.003):
    return abs(z1 - z2) <= tolerance

def validate_plan(plan):
    blocks = plan["blocks"]
    collisions = []

    for i in range(len(blocks)):
        pos_i = blocks[i]["goal_position"]
        quat_i = blocks[i]["goal_quaternion_wxyz"]
        yaw_i = quat_wxyz_to_yaw(quat_i)

        poly_i = rectangle_corners_xy(pos_i, yaw_i)

        for j in range(i + 1, len(blocks)):
            pos_j = blocks[j]["goal_position"]
            quat_j = blocks[j]["goal_quaternion_wxyz"]
            yaw_j = quat_wxyz_to_yaw(quat_j)

            # only care if their heights overlap
            if not same_layer(pos_i[2], pos_j[2]):
                continue

            poly_j = rectangle_corners_xy(pos_j, yaw_j)

            if polygons_overlap(poly_i, poly_j):
                collisions.append({
                    "block_a": i,
                    "block_b": j,
                    "pos_a": pos_i,
                    "pos_b": pos_j,
                    "quat_a": quat_i,
                    "quat_b": quat_j,
                    "message": f"Block {i} collides with block {j}"
                })

    return {
        "valid": len(collisions) == 0,
        "collisions": collisions,
    }