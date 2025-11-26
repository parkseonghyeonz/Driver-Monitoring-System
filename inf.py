from ultralytics import YOLO

model = YOLO("/content/runs/detect/train2/weights/best8.pt")

img_path = "/content/datasets/coco8/images/val/000000000036.jpg"

results = model(img_path, save=True)  

results[0].show()    
