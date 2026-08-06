import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from rfdetr import RFDETRMedium

if __name__ == '__main__':
    model = RFDETRMedium()

    model.train(
        dataset_dir="/home/xisai/Workspace/lalain/frosch/Frosch bottle 5.v6i.coco-segmentation",
        epochs=50,
        batch_size=4,
        num_workers=4,
        output_dir="runs/frosch_medium"
    )
