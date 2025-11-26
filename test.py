from ultralytics import YOLO
import cv2

model = YOLO(r'C:\DMS\model\helmet_best.pt')

cap = cv2.VideoCapture(r"C:\DMS\data\test_video8.mp4")
cv2.namedWindow('show', cv2.WINDOW_NORMAL)
cv2.resizeWindow('show', 800, 600)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, classes=[0,2,4,7], save=False, verbose=False, conf=0.5, iou=0.6)

    annotated_frame = results[0].plot()

    cv2.imshow('show', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()