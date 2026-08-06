from rfdetr import RFDETRMedium

if __name__ == '__main__':
    model = RFDETRMedium(
        pretrain_weights="runs/frosch_medium/checkpoint_best_regular.pth"
    )
    model.export(output_dir="runs/frosch_medium")
    print("ONNX export done.")
