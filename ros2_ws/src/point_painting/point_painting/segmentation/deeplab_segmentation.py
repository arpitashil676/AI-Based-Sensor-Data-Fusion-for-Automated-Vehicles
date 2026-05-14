import numpy as np
import cv2
from PIL import Image


def load_model(checkpoint_path: str = None):
    """
    Load YOLO26n-seg model. Auto-downloads on first use (~6 MB, cached).
    Pass checkpoint_path to override with a local file.
    """
    from ultralytics import YOLO
    model_path = checkpoint_path if checkpoint_path else 'yolo26n-seg.pt'
    return YOLO(model_path)


def segment_image(model, image: Image.Image) -> np.ndarray:
    """
    Run YOLO instance segmentation on a PIL image.
    Returns a (H, W) array with native YOLO/COCO class IDs per pixel, 0 = background.

    COCO class IDs (relevant for driving):
      0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
    """
    img_np = np.array(image)
    h, w = img_np.shape[:2]

    results = model(img_np, verbose=False, conf=0.15)
    label_mask = np.full((h, w), -1, dtype=np.int32)  # -1 = background/no detection

    for result in results:
        if result.masks is None:
            continue

        masks = result.masks.data.cpu().numpy()        # (N, H', W')
        classes = result.boxes.cls.cpu().numpy().astype(int)  # (N,) — native COCO IDs

        for mask, cls_id in zip(masks, classes):
            mask_u8 = (mask * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
            label_mask[mask_resized > 127] = cls_id

    return label_mask


if __name__ == '__main__':
    import argparse
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Run YOLO segmentation on an image')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--checkpoint', default=None, help='Path to YOLO model file (default: yolo26n-seg.pt)')
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    image = Image.open(args.image).convert('RGB')
    label_mask = segment_image(model, image)

    print('Detected classes:', np.unique(label_mask[label_mask > 0]))
    print('Class names:', {i: model.names[i] for i in np.unique(label_mask) if i in model.names})

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.imshow(image); plt.title('Original'); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(label_mask, cmap='tab20'); plt.title('YOLO Classes'); plt.axis('off')
    out = args.image.rsplit('.', 1)[0] + '_yolo_seg.png'
    plt.savefig(out)
    print(f'Saved: {out}')
