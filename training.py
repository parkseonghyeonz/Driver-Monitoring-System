from ultralytics import YOLO

def main():
    model = YOLO("yolo12s.pt")  # 또는 yolov12m.pt

    model.train(
        data=r"C:\DMS\hardhat_dataset\data.yaml",
        epochs=80,
        imgsz=640,
        batch=4,
        patience=12
    )

if __name__ == "__main__":
    main()
