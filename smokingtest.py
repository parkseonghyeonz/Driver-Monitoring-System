import cv2, math, time
from ultralytics import YOLO
import mediapipe as mp


ciga_model = YOLO(r'model\ciga_best.pt')
face_model = YOLO(r'model\yolov12m-face.pt')

ciga_model.to("cuda")
face_model.to("cuda")


# face mesh 초기화
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)



cap = cv2.VideoCapture(r"C:\DMS\data\smoke_test6.mp4")



cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 800, 600)


def get_frame():
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def face_detect_and_expand(frame):
    h, w = frame.shape[:2]
    face_result = face_model.predict(frame, conf=0.5, iou=0.5, verbose=False)[0]

    if len(face_result.boxes) == 0:
        return None

    x1, y1, x2, y2 = map(int, face_result.boxes[0].xyxy[0])

    center_x = (x1 + x2) /2
    center_y = (y1+y2)/2

    w=x2-x1
    h=y2-y1


    # 얼굴 bbox 1.5배 확장
    scale = 1.5

    new_w = w*scale
    new_h = h*scale

    new_x1 = int(center_x - new_w /2)
    new_y1 = int(center_y - new_h/2)
    new_x2=int(center_x+new_w/2)
    new_y2 = int(center_y+new_h/2)

    return new_x1, new_y1, new_x2, new_y2

def detect_ciga(frame, new_x1, new_y1, new_x2, new_y2):

    roi = frame[new_y1:new_y2, new_x1:new_x2]

    ciga_result = ciga_model.predict(roi, conf=0.6, iou=0.5, verbose=False)[0]
    if len(ciga_result.boxes)==0:
        return None
    ciga_x1, ciga_y1, ciga_x2, ciga_y2 = map(int, ciga_result.boxes[0].xyxy[0])

    # roi좌표 -> 원래 frame 좌표로 변환
    ciga_x1 = ciga_x1 + new_x1
    ciga_y1 = ciga_y1 + new_y1
    ciga_x2 = ciga_x2 + new_x1
    ciga_y2 = ciga_y2 + new_y1

    return ciga_x1, ciga_y1, ciga_x2, ciga_y2


frame_count = 0
last_face_bbox = None

while True:
    frame = get_frame()
    if frame is None:
        break

    annotated = frame.copy()

    if frame_count % 8 ==0:

        bbox = face_detect_and_expand(frame)

        if bbox is not None:
            last_face_bbox = bbox
    frame_count +=1

    if last_face_bbox is None:
        cv2.imshow("frame", annotated)
        if cv2.waitKey(1) == ord('q'):
            break
        continue

    new_x1, new_y1, new_x2, new_y2 = last_face_bbox

    cv2.rectangle(annotated, (new_x1, new_y1), (new_x2, new_y2), (0, 0, 255), 4)
    
    ciga_bbox = detect_ciga(frame,new_x1, new_y1, new_x2, new_y2)

    if ciga_bbox is not None:
        ciga_x1, ciga_y1, ciga_x2, ciga_y2 = ciga_bbox
        cv2.rectangle(annotated, (ciga_x1, ciga_y1), (ciga_x2, ciga_y2), (255,0,0), 2)



    cv2.imshow("frame", annotated)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
