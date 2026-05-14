"""
Pipeline isolation test — no ROS, no live node needed.

What this does:
  1. Extracts one paired image + LiDAR frame from the bag
  2. Runs YOLO segmentation on the image
  3. Projects LiDAR points onto the image using calib.txt
  4. Colours each projected point by its class
  5. Saves 4 images so you can visually verify each stage

Run inside the container:
  python3 /workspace/test_pipeline_isolation.py

Output (saved to /workspace/isolation_output/):
  01_raw_image.jpg          — original camera frame
  02_yolo_mask.jpg          — YOLO class mask (colour per class)
  03_overlay.jpg            — mask blended on top of image
  04_lidar_projected.jpg    — LiDAR points drawn ON the image, coloured by class
"""

import sys
import os
import numpy as np
import cv2

# ── Paths ──────────────────────────────────────────────────────────────────────
BAG_PATH   = '/workspace/studentProject1/'
CALIB_PATH = '/workspace/calib.txt'
OUTPUT_DIR = '/workspace/isolation_output'
YOLO_MODEL = 'yolo26n-seg.pt'

# COCO class colours (BGR for OpenCV): person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7
CLASS_COLORS_BGR = {
    0:  (0,   0,   255),   # person     — red
    1:  (255, 0,   0),     # bicycle    — blue
    2:  (0,   255, 0),     # car        — green
    3:  (0,   128, 255),   # motorcycle — orange
    5:  (0,   255, 255),   # bus        — yellow
    7:  (0,   200, 0),     # truck      — dark green
}
UNPAINTED_BGR = (80, 80, 80)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Stage 1: Extract one paired frame from the bag ─────────────────────────────
print('\n[Stage 1] Extracting one frame from bag...')

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2 as pc2

reader = rosbag2_py.SequentialReader()
reader.open(
    rosbag2_py.StorageOptions(uri=BAG_PATH, storage_id='sqlite3'),
    rosbag2_py.ConverterOptions(input_serialization_format='cdr',
                                output_serialization_format='cdr')
)
topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
bridge = CvBridge()

img_frame = None
lidar_frame = None

while reader.has_next() and (img_frame is None or lidar_frame is None):
    topic, data, _ = reader.read_next()

    if topic == '/blackfly_s/cam0/image_rectified' and img_frame is None:
        msg = deserialize_message(data, get_message(topic_types[topic]))
        img_frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        print(f'  Got image: {img_frame.shape}')

    elif topic == '/velodyne/points_raw' and lidar_frame is None:
        msg = deserialize_message(data, get_message(topic_types[topic]))
        pts = list(pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True))
        lidar_frame = np.array([(p[0], p[1], p[2]) for p in pts], dtype=np.float32)
        print(f'  Got LiDAR: {lidar_frame.shape[0]} points')

cv2.imwrite(f'{OUTPUT_DIR}/01_raw_image.jpg', img_frame)
print(f'  Saved: 01_raw_image.jpg')


# ── Stage 2: Run YOLO segmentation ────────────────────────────────────────────
print('\n[Stage 2] Running YOLO segmentation...')

from ultralytics import YOLO

model = YOLO(YOLO_MODEL)
img_rgb = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)
h, w = img_frame.shape[:2]

results = model(img_rgb, verbose=False)
label_mask = np.full((h, w), -1, dtype=np.int32)  # -1 = background/no detection

for result in results:
    if result.masks is None:
        continue
    masks   = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    for mask, cls_id in zip(masks, classes):
        mask_u8 = (mask * 255).astype(np.uint8)
        mask_resized = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
        label_mask[mask_resized > 127] = cls_id

# Visualise mask
detected = np.unique(label_mask[label_mask >= 0])
print(f'  Detected classes: {[(c, model.names[c]) for c in detected if c in model.names]}')

mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
for cls_id, color in CLASS_COLORS_BGR.items():
    mask_vis[label_mask == cls_id] = color

cv2.imwrite(f'{OUTPUT_DIR}/02_yolo_mask.jpg', mask_vis)
print(f'  Saved: 02_yolo_mask.jpg')


# ── Stage 3: Overlay mask on image ────────────────────────────────────────────
print('\n[Stage 3] Creating segmentation overlay...')

overlay = cv2.addWeighted(img_frame, 0.6, mask_vis, 0.4, 0)

# Add class legend
y = 30
for cls_id, color in CLASS_COLORS_BGR.items():
    if cls_id in detected:
        name = model.names.get(cls_id, str(cls_id))
        cv2.rectangle(overlay, (10, y-15), (30, y+5), color, -1)
        cv2.putText(overlay, name, (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y += 30

cv2.imwrite(f'{OUTPUT_DIR}/03_overlay.jpg', overlay)
print(f'  Saved: 03_overlay.jpg')


# ── Stage 4: Project LiDAR points onto image ──────────────────────────────────
print('\n[Stage 4] Projecting LiDAR points onto image...')

sys.path.insert(0, '/workspace/ros2_ws/src/perception_framework')
sys.path.insert(0, '/workspace/ros2_ws/install/perception_framework/lib/python3.10/site-packages')

from perception_framework.lidar_to_image_projection import KittiLidarToImageProjector

projector = KittiLidarToImageProjector(CALIB_PATH)
image_points, valid_lidar = projector.project_lidar_to_image(lidar_frame, (h, w))

print(f'  Total points: {len(lidar_frame)}')
print(f'  Points in camera frame: {len(image_points)} ({100*len(image_points)/len(lidar_frame):.1f}%)')

# Draw projected points coloured by their class from the mask
lidar_img = overlay.copy()
painted = 0
class_counts = {}

for pt_px in image_points:
    u, v = int(pt_px[0]), int(pt_px[1])
    if 0 <= u < w and 0 <= v < h:
        cls_id = label_mask[v, u]
        color = CLASS_COLORS_BGR.get(cls_id, UNPAINTED_BGR)
        cv2.circle(lidar_img, (u, v), radius=3, color=color, thickness=-1)
        if cls_id >= 0:
            painted += 1
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

print(f'  Painted points (non-background): {painted}')
for cls_id, count in sorted(class_counts.items()):
    name = model.names.get(cls_id, str(cls_id))
    print(f'    {name} (class {cls_id}): {count} points')

cv2.imwrite(f'{OUTPUT_DIR}/04_lidar_projected.jpg', lidar_img)
print(f'  Saved: 04_lidar_projected.jpg')

print(f'\nDone. Open /workspace/isolation_output/ to see all 4 images.')
print('If 04_lidar_projected.jpg shows coloured dots on the car and person,')
print('the full pipeline is correct.')
