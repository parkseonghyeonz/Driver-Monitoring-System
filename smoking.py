# # yolo + facemesh 기반 흡연 탐지
# import cv2, math, time
# import mediapipe as mp
# from ultralytics import YOLO

# mp_face = mp.solutions.face_mesh
# face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)

# model = YOLO(r'model\best.pt')  
# face_model = YOLO(r'model\yolov12s-face.pt')

# model.to("cuda")
# face_model.to("cuda")

# cap = cv2.VideoCapture(r"C:\DMS\data\smoke_test6.mp4")

# cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('frame', 800, 600)

# smoking = 0
# prev_smoking = 0
# last_smoke_time = 0.0

# frame_count = 0
# cached_face = None
# cached_smoke = None
# cached_mesh = None
# cached_roi_box = None


# # ==================================================
# # 1) 얼굴 검출 + ROI 확장 + Mediapipe FaceMesh 실행
# # ==================================================
# def detect_face_and_expand(frame):
#     h, w = frame.shape[:2]

#     face_result = face_model.predict(
#         frame, save=False, verbose=False,
#         conf=0.4, iou=0.5, device=0, imgsz=320
#     )

#     if not face_result or len(face_result[0].boxes) == 0:
#         return None, None, None

#     box = face_result[0].boxes[0]
#     x1, y1, x2, y2 = map(int, box.xyxy[0])

#     face_crop = frame[y1:y2, x1:x2]
#     mesh_res = face_mesh.process(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
#     if not mesh_res.multi_face_landmarks:
#         return None, None, None

#     face_w = x2 - x1
#     face_h = y2 - y1
#     cx_face = (x1 + x2) // 2
#     cy_face = (y1 + y2) // 2

#     scale = 1.5
#     new_w = int(face_w * scale)
#     new_h = int(face_h * scale)

#     ex1 = max(0, cx_face - new_w // 2)
#     ey1 = max(0, cy_face - new_h // 2)
#     ex2 = min(w, cx_face + new_w // 2)
#     ey2 = min(h, cy_face + new_h // 2)

#     expand_crop = frame[ey1:ey2, ex1:ex2]

#     return expand_crop, (ex1, ey1, ex2, ey2), mesh_res


# # ==================================================
# # 2) 밑입술 좌표 계산
# # ==================================================
# def low_lip(mesh_res, box):
#     ex1, ey1, ex2, ey2 = box
#     lm = mesh_res.multi_face_landmarks[0].landmark[14]
#     lip_x = int(lm.x * (ex2 - ex1)) + ex1
#     lip_y = int(lm.y * (ey2 - ey1)) + ey1
#     return lip_x, lip_y


# # ==================================================
# # 3) 담배 탐지
# # ==================================================
# def cigarette_detect(expand_crop):
#     return model.predict(
#         expand_crop, save=False, verbose=False,
#         conf=0.5, iou=0.5, device=0
#     )[0]


# # ==================================================
# # 4) 거리 계산
# # ==================================================
# def dist_def(cx, cy, lip_x, lip_y):
#     return math.dist((cx, cy), (lip_x, lip_y))


# # ==================================================
# # 메인 루프
# # ==================================================
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     annotated = frame.copy()
#     frame_count += 1

#     # ================================
#     # 5프레임마다 얼굴 + FaceMesh 실행
#     # ================================
#     if frame_count % 3 == 0:
#         detect_result = detect_face_and_expand(frame)

#         if detect_result[0] is not None:
#             cached_face, cached_roi_box, cached_mesh = detect_result
#         else:
#             cached_face = None

#     # 얼굴 없으면 넘어감
#     if cached_face is None:
#         cv2.imshow("frame", annotated)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#         continue

#     expand_crop = cached_face
#     ex1, ey1, ex2, ey2 = cached_roi_box

#     cv2.rectangle(annotated, (ex1, ey1), (ex2, ey2), (255, 255, 0), 2)

#     # 밑입술 좌표
#     lip_x, lip_y = low_lip(cached_mesh, cached_roi_box)
#     cv2.circle(annotated, (lip_x, lip_y), 4, (0, 255, 255), -1)

#     # ================================
#     # 5프레임마다 담배 탐지 실행
#     # ================================
#     if frame_count % 3 == 0:
#         cached_smoke = cigarette_detect(expand_crop)

#     smoke_results = cached_smoke

#     # ================================
#     # 담배 bbox + 거리 계산
#     # ================================
#     if smoke_results and len(smoke_results.boxes) > 0:
#         for box in smoke_results.boxes:
#             sx1, sy1, sx2, sy2 = map(int, box.xyxy[0])

#             x1_obj = sx1 + ex1
#             y1_obj = sy1 + ey1
#             x2_obj = sx2 + ex1
#             y2_obj = sy2 + ey1

#             cx = (x1_obj + x2_obj) // 2
#             cy = (y1_obj + y2_obj) // 2

#             cv2.rectangle(annotated, (x1_obj, y1_obj), (x2_obj, y2_obj), (255, 0, 0), 2)

#             d = dist_def(cx, cy, lip_x, lip_y)

#             if d < 30:
#                 smoking = 1
#                 last_smoke_time = time.time()

#     if smoking == 1 and time.time() - last_smoke_time > 10:
#         smoking = 0

#     if smoking != prev_smoking:
#         print("흡연 중" if smoking else "흡연 아님")
#         prev_smoking = smoking

#     if smoking:
#         cv2.putText(annotated, "!! SMOKING !!", (50, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

#     cv2.imshow("frame", annotated)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# yolo + opencv traker + facemesh 기반 흡연 탐지
# import cv2, math, time
# import mediapipe as mp
# from ultralytics import YOLO

# # ================================
# # 초기화
# # ================================
# mp_face = mp.solutions.face_mesh
# face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)

# face_model = YOLO('model/yolov12s-face.pt')
# smoke_model = YOLO('model/best.pt')

# face_model.to("cuda")
# smoke_model.to("cuda")

# cap = cv2.VideoCapture(r"C:\DMS\data\smoke_test6.mp4")

# cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('frame', 800, 600)

# # 상태 변수
# tracker = None
# face_box = None
# frame_count = 0

# cached_mesh = None
# cached_smoke = None
# cached_roi = None
# expand_crop = None

# smoking = 0
# prev_smoking = 0
# last_smoke_time = 0

# # ================================
# # YOLO 얼굴 검출
# # ================================
# def detect_face_yolo(frame):
#     res = face_model.predict(frame, conf=0.4, iou=0.5, imgsz=320, verbose=False)[0]
#     if len(res.boxes) == 0:
#         return None
#     return tuple(map(int, res.boxes[0].xyxy[0]))  # (x1,y1,x2,y2)

# # ================================
# # ROI 확장 + FaceMesh
# # ================================
# def expand_and_mesh(frame, box):
#     h, w = frame.shape[:2]
#     x1, y1, x2, y2 = box

#     cx = (x1 + x2) // 2
#     cy = (y1 + y2) // 2
#     fw, fh = x2 - x1, y2 - y1

#     scale = 1.5
#     exw, exh = int(fw * scale), int(fh * scale)

#     ex1 = max(0, cx - exw // 2)
#     ey1 = max(0, cy - exh // 2)
#     ex2 = min(w, cx + exw // 2)
#     ey2 = min(h, cy + exh // 2)

#     crop = frame[ey2:ey1, ex2:ex1]
#     crop = frame[ey1:ey2, ex1:ex2]

#     mesh = face_mesh.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

#     if not mesh.multi_face_landmarks:
#         return None, None, None

#     return crop, (ex1, ey1, ex2, ey2), mesh

# # ================================
# # 밑입술 좌표
# # ================================
# def get_low_lip(mesh, roi_box):
#     ex1, ey1, ex2, ey2 = roi_box
#     lm = mesh.multi_face_landmarks[0].landmark[14]

#     lx = int(lm.x * (ex2 - ex1)) + ex1
#     ly = int(lm.y * (ey2 - ey1)) + ey1
#     return lx, ly

# # ================================
# # CSRT 트래커 초기화
# # ================================
# def init_tracker(frame, box):
#     tracker = cv2.TrackerCSRT_create()
#     x1, y1, x2, y2 = box
#     tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
#     return tracker

# # ================================
# # 메인 루프
# # ================================
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1
#     annotated = frame.copy()

#     # ================================
#     # 1) 3프레임마다 YOLO 얼굴 검출 → 트래커 리셋
#     # ================================
#     if frame_count % 5 == 0:
#         new_box = detect_face_yolo(frame)
#         if new_box:
#             face_box = new_box
#             tracker = init_tracker(frame, new_box)

#     # 트래커 실패 시 다음 프레임에서 YOLO로 복구
#     if tracker is None:
#         cv2.imshow("frame", annotated)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#         continue

#     success, tbox = tracker.update(frame)
#     if not success:
#         tracker = None
#         continue

#     # tracker box → face_box 업데이트
#     x, y, w, h = map(int, tbox)
#     face_box = (x, y, x + w, y + h)

#     cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 255, 0), 2)

#     # ================================
#     # 2) 3프레임마다 FaceMesh + ROI 확장
#     # ================================
#     if frame_count % 5 == 0:
#         crop, roi, mesh = expand_and_mesh(frame, face_box)
#         if roi is not None:
#             expand_crop = crop
#             cached_roi = roi
#             cached_mesh = mesh

#     if cached_mesh is None:
#         cv2.imshow("frame", annotated)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#         continue

#     # 밑입술 그리기
#     lip_x, lip_y = get_low_lip(cached_mesh, cached_roi)
#     cv2.circle(annotated, (lip_x, lip_y), 4, (0, 255, 255), -1)

#     # ================================
#     # 3) 3프레임마다 담배 탐지
#     # ================================
#     if frame_count % 5 == 0:
#         cached_smoke = smoke_model.predict(expand_crop, conf=0.5, iou=0.5, verbose=False)[0]

#     # ================================
#     # 4) 담배 위치 + 입 거리 계산
#     # ================================
#     if cached_smoke is not None and len(cached_smoke.boxes) > 0:
#         for b in cached_smoke.boxes:
#             sx1, sy1, sx2, sy2 = map(int, b.xyxy[0])
#             ex1, ey1, _, _ = cached_roi

#             # 좌표를 전체 프레임 좌표로 변환
#             gx1, gy1 = sx1 + ex1, sy1 + ey1
#             gx2, gy2 = sx2 + ex1, sy2 + ey1
#             cx = (gx1 + gx2) // 2
#             cy = (gy1 + gy2) // 2

#             cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)

#             # 거리 판단
#             d = math.dist((cx, cy), (lip_x, lip_y))
#             if d < 30:
#                 smoking = 1
#                 last_smoke_time = time.time()

#     # ================================
#     # 5) 흡연 상태 유지
#     # ================================
#     if smoking == 1 and time.time() - last_smoke_time > 10:
#         smoking = 0

#     if smoking != prev_smoking:
#         print("흡연 중" if smoking else "흡연 아님")
#         prev_smoking = smoking

#     if smoking:
#         cv2.putText(annotated, "!! SMOKING !!", (50, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

#     cv2.imshow("frame", annotated)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# No face mesh, yolo bbox 안에 담배가 있는지로 판단
# import cv2, math, time
# from ultralytics import YOLO

# # ================================
# # 모델 로드
# # ================================
# ciga_model = YOLO(r'model\ciga_best.pt')
# face_model  = YOLO(r'model\yolov12s-face.pt')

# ciga_model.to("cuda")
# face_model.to("cuda")

# cap = cv2.VideoCapture(r"C:\DMS\data\smoke_test6.mp4")

# cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('frame', 800, 600)

# smoking = 0
# prev_smoking = 0
# last_smoke_time = 0.0

# frame_count = 0
# cached_roi_box = None


# def face_detect_and_expand(frame):
#     h, w = frame.shape[:2]

#     face_result = face_model.predict(frame, conf=0.4, iou=0.5,verbose=False)[0]

#     if len(face_result.boxes) == 0:
#         return None

#     x1, y1, x2, y2 = map(int, face_result.boxes[0].xyxy[0])

#     cx = (x1 + x2) // 2
#     cy = (y1 + y2) // 2
#     fw = x2 - x1
#     fh = y2 - y1

#     scale = 1.5
#     nw = int(fw * scale)
#     nh = int(fh * scale)

#     ex1 = max(0, cx - nw // 2)
#     ey1 = max(0, cy - nh // 2)
#     ex2 = min(w, cx + nw // 2)
#     ey2 = min(h, cy + nh // 2)

#     return (ex1, ey1, ex2, ey2)

# def ciga_detect(frame, roi_box):
#     ex1, ey1, ex2, ey2 = roi_box
#     crop = frame[ey1:ey2, ex1:ex2]

#     ciga_detect = ciga_model.predict(crop, conf=0.5, iou=0.5,verbose=False)[0]

#     return ciga_detect


# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     annotated = frame.copy()
#     frame_count += 1
#     now = time.time()

#     if frame_count % 3 == 0:
#         roi_box = face_detect_and_expand(frame)
#         if roi_box is not None:
#             cached_roi_box = roi_box
#         else:
#             cached_roi_box = None

#     if cached_roi_box is None:
#         cv2.imshow("frame", annotated)
#         if cv2.waitKey(1) == ord('q'):
#             break
#         continue

#     ex1, ey1, ex2, ey2 = cached_roi_box
#     cv2.rectangle(annotated, (ex1, ey1), (ex2, ey2), (255, 255, 0), 2)

#     smoke_res = ciga_detect(frame, cached_roi_box)

#     ciga_state = False

#     if smoke_res and len(smoke_res.boxes) > 0:
#         for box in smoke_res.boxes:
#             sx1, sy1, sx2, sy2 = map(int, box.xyxy[0])

#             gx1 = sx1 + ex1
#             gy1 = sy1 + ey1
#             gx2 = sx2 + ex1
#             gy2 = sy2 + ey1

#             cx = (gx1 + gx2) // 2
#             cy = (gy1 + gy2) // 2

#             cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)

#             if ex1 < cx < ex2 and ey1 < cy < ey2:
#                 ciga_state = True
#                 last_smoke_time = now

#     if ciga_state:
#         smoking = 1
#     else:
#         if smoking == 1 and (now - last_smoke_time > 10):
#             smoking = 0

#     if smoking != prev_smoking:
#         print("흡연 중" if smoking else "흡연 아님")
#         prev_smoking = smoking

#     cv2.imshow("frame", annotated)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



import cv2, math, time
from ultralytics import YOLO
import mediapipe as mp


ciga_model = YOLO(r'model\ciga_best.pt')
face_model = YOLO(r'model\yolov12s-face.pt')

ciga_model.to("cuda")
face_model.to("cuda")


# face mesh 초기화
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(r"C:\DMS\data\smoke_test6.mp4")

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 800, 600)

smoking = 0 # 담배 피는지 여부  
prev_smoking = 0 # 이전 프레임 담배 피는지
last_smoke_time = 0 # 마지막 흡연 시간

frame_count = 0 
cached_roi_box = None # 얼굴 확장 박스 좌표
cached_lip = None   # 밑입술 좌표


def face_detect_and_expand(frame):
    h, w = frame.shape[:2]
    face_result = face_model.predict(frame, conf=0.4, iou=0.5, verbose=False)[0]

    if len(face_result.boxes) == 0:
        return None, None

    x1, y1, x2, y2 = map(int, face_result.boxes[0].xyxy[0])

    # 얼굴 중심 계산
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    fw = x2 - x1
    fh = y2 - y1

    # 얼굴 bbox 1.5배 확장
    scale = 1.5
    nw = int(fw * scale)
    nh = int(fh * scale)

    ex1 = max(0, cx - nw // 2)
    ey1 = max(0, cy - nh // 2)
    ex2 = min(w, cx + nw // 2)
    ey2 = min(h, cy + nh // 2)

    return (ex1, ey1, ex2, ey2), frame[ey1:ey2, ex1:ex2]


def lip_center(face_crop, roi_box):
    ex1, ey1, ex2, ey2 = roi_box
    crop_h, crop_w = face_crop.shape[:2]

    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    mesh = face_mesh.process(rgb)

    if not mesh.multi_face_landmarks:
        return None

    lm = mesh.multi_face_landmarks[0].landmark[14]

    lx = int(lm.x * crop_w) + ex1
    ly = int(lm.y * crop_h) + ey1

    return (lx, ly)


# 확장한 bbox 내에서 담배 탐지
def ciga_detect(frame, roi_box):
    ex1, ey1, ex2, ey2 = roi_box
    crop = frame[ey1:ey2, ex1:ex2]
    ciga_result = ciga_model.predict(crop, conf=0.6, iou=0.6, verbose=False)
    return ciga_result


while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated = frame.copy()
    frame_count += 1
    now = time.time()

    # 3프레임마다 얼굴 검출 + ROI 확장 + 밑입술 좌표 계산 
    if frame_count % 3 == 0:
        roi_box, face_crop = face_detect_and_expand(frame)

        if roi_box is not None and face_crop is not None:
            cached_roi_box = roi_box

            lip = lip_center(face_crop, roi_box)
            if lip is not None:
                cached_lip = lip
        else:
            cached_roi_box = None


    if cached_roi_box is None or cached_lip is None:
        cv2.imshow("frame", annotated)
        if cv2.waitKey(1) == ord('q'):
            break
        continue

    ex1, ey1, ex2, ey2 = cached_roi_box
    lip_x, lip_y = cached_lip



    cv2.rectangle(annotated, (ex1, ey1), (ex2, ey2), (255, 255, 0), 2)
    cv2.circle(annotated, (lip_x, lip_y), 4, (0, 255, 255), -1)


    # 담배는 매프레임마다 탐지
    smoke_res = ciga_detect(frame, cached_roi_box)
    smoke_res = smoke_res[0]

    detected = False

    if smoke_res and len(smoke_res.boxes) > 0:
        for box in smoke_res.boxes:
            sx1, sy1, sx2, sy2 = map(int, box.xyxy[0])


            gx1 = sx1 + ex1
            gy1 = sy1 + ey1
            gx2 = sx2 + ex1
            gy2 = sy2 + ey1

            cx = (gx1 + gx2) // 2
            cy = (gy1 + gy2) // 2

            cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)

            d = math.dist((cx, cy), (lip_x, lip_y))
            face_width = ex2-ex1
            threshold = face_width * 0.25



            if d < threshold: 
                detected = True
                last_smoke_time = now

    if detected:
        smoking = 1
    else:
        if smoking == 1 and (now - last_smoke_time > 10):
            smoking = 0

    if smoking != prev_smoking:
        if smoking:
            print("흡연 중 ")
        else:
            print("흡연 아님")

        prev_smoking = smoking


    cv2.imshow("frame", annotated)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
