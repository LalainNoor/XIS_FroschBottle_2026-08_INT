from rfdetr import RFDETRMedium, RFDETRSegMedium

CHECKPOINT = "runs/frosch_medium/checkpoint_best_regular.pth"
SEG_CHECKPOINT = "runs/frosch_seg_medium/checkpoint_best_total.pth"

print("Loading detection model...")
model = RFDETRMedium(pretrain_weights=CHECKPOINT)

print("Exporting detection model to TensorRT...")
model.export(
    format="tensorrt",
    fp16=True,
)

print("Loading segmentation model...")
seg_model = RFDETRSegMedium(pretrain_weights=SEG_CHECKPOINT)

print("Exporting segmentation model to TensorRT...")
seg_model.export(
    format="tensorrt",
    fp16=True,
)

print("TensorRT export completed.")

