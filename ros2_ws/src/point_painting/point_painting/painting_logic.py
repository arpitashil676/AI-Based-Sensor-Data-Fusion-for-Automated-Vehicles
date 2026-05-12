import numpy as np
from perception_framework.lidar_to_image_projection import KittiLidarToImageProjector

_projector: KittiLidarToImageProjector | None = None


def init_projector(calib_file_path: str) -> None:
    global _projector
    _projector = KittiLidarToImageProjector(calib_file_path)


def paint_points(points_xyz: np.ndarray, seg_image: np.ndarray):
    """
    Project each LiDAR point onto the segmentation mask and return its class ID.

    Returns (painted_count, skipped_count, class_ids).
    class_ids[i] == -1 means point i did not land inside the image.
    """
    n = len(points_xyz)

    if _projector is None or n == 0:
        return 0, n, [-1] * n

    h, w = seg_image.shape[:2]

    # Single projection pass — get pixel coords for all in-frame points
    image_points, valid_lidar = _projector.project_lidar_to_image(points_xyz, (h, w))

    if len(image_points) == 0:
        return 0, n, [-1] * n

    # Find which original indices correspond to the in-frame points.
    # Project all points to camera space, keep depth>0, then apply same
    # in-bounds mask to recover original indices.
    camera_pts = _projector.lidar_to_camera(points_xyz)
    depth_ok = camera_pts[:, 2] > 0
    depth_indices = np.where(depth_ok)[0]

    cam_depth_pts = camera_pts[depth_ok]
    import cv2
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    proj, _ = cv2.projectPoints(
        cam_depth_pts.astype(np.float64),
        rvec, tvec,
        _projector.camera_matrix.astype(np.float64),
        _projector.dist_coeffs,
    )
    proj = proj.reshape(-1, 2)
    u_all, v_all = proj[:, 0], proj[:, 1]
    inside = (u_all >= 0) & (u_all < w) & (v_all >= 0) & (v_all < h)

    orig_indices = depth_indices[inside]
    u_in = np.clip(u_all[inside].astype(int), 0, w - 1)
    v_in = np.clip(v_all[inside].astype(int), 0, h - 1)

    class_ids = np.full(n, -1, dtype=int)
    class_ids[orig_indices] = seg_image[v_in, u_in]
    painted = int(inside.sum())

    return painted, n - painted, class_ids.tolist()
