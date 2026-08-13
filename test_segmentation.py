import cv2
import numpy as np
from rfdetr import RFDETRSegMedium

CHECKPOINT = "runs/frosch_seg_medium/checkpoint_best_total.pth"
IMAGE = "/home/xisai/Workspace/lalain/frosch/Frosch bottle 5.v6i.coco-segmentation/test/frame_20260610_095942_771350.jpg"

model = RFDETRSegMedium(
    pretrain_weights=CHECKPOINT
)

image = cv2.imread(IMAGE)

if image is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE}")

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Lower threshold temporarily so we can see ALL predictions.
detections = model.predict(
    rgb,
    threshold=0.20
)

print("\n==============================")
print("SEGMENTATION DIAGNOSTIC")
print("==============================")

print("Number of detections:", len(detections.class_id))
print("Class IDs:", detections.class_id)
print("Confidences:", detections.confidence)

if hasattr(model, "class_names"):
    print("Model class names:", model.class_names)

if hasattr(detections, "data"):
    print("Detection data keys:", detections.data.keys())

print("==============================\n")

for i, class_id in enumerate(detections.class_id):

    confidence = float(detections.confidence[i])

    print(
        f"Detection {i}: "
        f"class_id={class_id}, "
        f"confidence={confidence:.3f}"
    )

    if detections.mask is None:
        continue

    mask = detections.mask[i]

    mask_uint8 = (
        mask.astype(np.uint8) * 255
    )

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        continue

    contour = max(
        contours,
        key=cv2.contourArea
    )

    # Draw every segmentation mask in green.
    cv2.drawContours(
        image,
        [contour],
        -1,
        (0, 255, 0),
        3
    )

    # Draw mask center.
    M = cv2.moments(contour)

    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        cv2.circle(
            image,
            (cx, cy),
            6,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            image,
            f"ID {class_id} {confidence:.2f}",
            (cx + 10, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    # Draw mask-based minimum-area rectangle.
    rect = cv2.minAreaRect(contour)

    box = cv2.boxPoints(rect)
    box = np.int32(box)

    cv2.drawContours(
        image,
        [box],
        -1,
        (255, 0, 0),
        2
    )


cv2.imwrite(
    "segmentation_diagnostic.jpg",
    image
)

print("\nSaved: segmentation_diagnostic.jpg")
