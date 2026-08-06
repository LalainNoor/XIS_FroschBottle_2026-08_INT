import easyocr

import cv2

reader = easyocr.Reader(['en'])
image_path ="/home/xisai/Workspace/lalain/frosch/Frosch bottle 5.v6i.coco-segmentation/test/frame_20260610_095942_771350.jpg"

results = reader.readtext(image_path)

for (bbox, text, confidence) in results:

    print(f"Text: {text} | Confidence: {confidence:.2f}")
