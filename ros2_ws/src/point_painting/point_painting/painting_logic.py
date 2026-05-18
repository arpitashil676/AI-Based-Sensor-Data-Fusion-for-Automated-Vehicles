"""
PointPainting core logic — projection and class label lookup.

This module is intentionally kept free of ROS dependencies so it can be
imported and unit-tested on any machine without a ROS runtime.

The two public functions are:
    init_projector(calib_file_path) — load calibration matrices once at startup
    paint_points(points_xyz, seg_image) — project LiDAR points and return class IDs

The projection uses KittiLidarToImageProjector from the perception_framework
package (implemented by arpitashil676). We call its lidar_to_camera() method
and camera_matrix directly — one projection pass, no duplication.
"""

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

    Uses KittiLidarToImageProjector (perception_framework) as the single
    source of projection — one projection pass, no duplicate math.
    """
    n = len(points_xyz)

    if _projector is None or n == 0:
        return 0, n, [-1] * n

    h, w = seg_image.shape[:2]

    # Step 1: transform all points to camera space (colleague's method)
    camera_pts = _projector.lidar_to_camera(points_xyz)

    # Step 2: keep only points in front of the camera
    # (points behind the camera would project to negative depth and cause issues)
    depth_ok = camera_pts[:, 2] > 0
    depth_indices = np.where(depth_ok)[0]
    cam_front = camera_pts[depth_ok]

    if len(cam_front) == 0:
        return 0, n, [-1] * n

    # Step 3: project to pixels using colleague's camera matrix
    import cv2
    proj, _ = cv2.projectPoints(
        cam_front.astype(np.float64),
        np.zeros((3, 1)), np.zeros((3, 1)),
        _projector.camera_matrix.astype(np.float64),
        _projector.dist_coeffs,
    )
    proj = proj.reshape(-1, 2)
    u_all, v_all = proj[:, 0], proj[:, 1]

    # Step 4: keep only points inside the image frame
    inside = (u_all >= 0) & (u_all < w) & (v_all >= 0) & (v_all < h)
    orig_indices = depth_indices[inside]
    u_in = np.clip(u_all[inside].astype(int), 0, w - 1)
    v_in = np.clip(v_all[inside].astype(int), 0, h - 1)

    # Step 5: look up class at each pixel
    class_ids = np.full(n, -1, dtype=int)
    class_ids[orig_indices] = seg_image[v_in, u_in]
    painted = int(inside.sum())

    return painted, n - painted, class_ids.tolist()
