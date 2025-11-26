# 비디오에서 프레임 읽고
# 8프레임마다 사람을 YOLO로 검출
# 사람 bbox를 crop하여 PPE YOLO 모델로 검사
# PPE 상태가 5프레임 이상 연속 동일하면 확정 (디바운싱)
# 화면에 bbox와 텍스트 출력

from ultralytics import YOLO
import cv2

model_person = YOLO(r"C:\DMS\model\yolo12m.pt")
model_ppe = YOLO(r"C:\DMS\model\ppe.pt")

cap = cv2.VideoCapture(r"C:\DMS\data\test_video1.mp4")

cv2.namedWindow('show', cv2.WINDOW_NORMAL)
cv2.resizeWindow('show', 800, 600)


def get_frame():
    ret, frame = cap.read()
    return frame if ret else None


# yolo - person detection
def person_detect(frame):
    person_result = model_person.predict(frame, classes=[0], conf=0.5, iou=0.5, verbose=False)[0]
    return person_result


# 사람 bbox 영역 crop
def expand_person_crop(frame, result):
    crops = []
    if result is None:
        return crops

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append((crop, (x1, y1, x2, y2)))

    return crops


# 프레임 카운터
frame_count = 0

# 캐시 저장 변수
last_person_result = None   # 8프레임마다 새로 갱신되는 결과 저장소
last_ppe_results = []       # PPE bbox 표시용 저장 리스트


# ------------------------------------------------------------
# 디바운싱용 상태값 저장
prev_state_helmet = None
prev_state_vest = None

helmet_candidate_state = None
vest_candidate_state = None

helmet_candidate_count = 0
vest_candidate_count = 0

maintain_frame = 5   # 상태 확정 최소 프레임
# ------------------------------------------------------------


while True:
    frame = get_frame()
    if frame is None:
        break

    # 8프레임마다 YOLO 재검출
    if frame_count % 8 == 0:

        # 1) 사람 detection
        last_person_result = person_detect(frame)

        # 2) PPE 결과 초기화
        last_ppe_results = []

        # 3) crop 생성
        crops = expand_person_crop(frame, last_person_result)

        # 상태 플래그 초기화
        any_hardhat = False
        any_nohardhat = False
        any_vest = False
        any_no_vest = False

        # 사람별 crop → PPE 분석
        for crop, bbox in crops:
            ppe_result = model_ppe.predict(
                crop, classes=[0, 2, 4, 7], conf=0.6, iou=0.6, verbose=False
            )[0]

            last_ppe_results.append((ppe_result, bbox))

            for hbox in ppe_result.boxes:
                class_name = model_ppe.names[int(hbox.cls[0])]

                if class_name == "Hardhat":
                    any_hardhat = True
                if class_name == "NO-Hardhat":
                    any_nohardhat = True
                if class_name == "Safety Vest":
                    any_vest = True
                if class_name == "NO-Safety Vest":
                    any_no_vest = True

        # raw 상태 계산
        if any_nohardhat:
            helmet_raw = "헬멧 미착용"
        elif any_hardhat:
            helmet_raw = "헬멧 착용"
        else:
            helmet_raw = "헬멧 감지안됨"

        if any_no_vest:
            vest_raw = "안전조끼 미착용"
        elif any_vest:
            vest_raw = "안전조끼 착용"
        else:
            vest_raw = "안전조끼 감지안됨"

        # ---- 디바운싱 ----
        # 헬멧
        if helmet_raw != "헬멧 감지안됨":
            if helmet_raw == helmet_candidate_state:
                helmet_candidate_count += 1
            else:
                helmet_candidate_state = helmet_raw
                helmet_candidate_count = 1

            if helmet_candidate_count >= maintain_frame and helmet_candidate_state != prev_state_helmet:
                print(helmet_candidate_state)
                prev_state_helmet = helmet_candidate_state

        # 조끼
        if vest_raw != "안전조끼 감지안됨":
            if vest_raw == vest_candidate_state:
                vest_candidate_count += 1
            else:
                vest_candidate_state = vest_raw
                vest_candidate_count = 1

            if vest_candidate_count >= maintain_frame and vest_candidate_state != prev_state_vest:
                print(vest_candidate_state)
                prev_state_vest = vest_candidate_state


    # ---- 화면 표시 ----
    annotated = frame.copy()

    if last_person_result is not None:
        annotated = last_person_result.plot(img=annotated)

    # PPE bbox 그리기
    for ppe_result, (x1, y1, x2, y2) in last_ppe_results:
        for hbox in ppe_result.boxes:
            hx1, hy1, hx2, hy2 = hbox.xyxy[0].cpu().numpy().astype(int)

            class_name = model_ppe.names[int(hbox.cls[0])]

            ox1 = x1 + hx1
            oy1 = y1 + hy1
            ox2 = x1 + hx2
            oy2 = y1 + hy2

            cv2.rectangle(annotated, (ox1, oy1), (ox2, oy2), (0, 0, 255), 4)
            cv2.putText(
                annotated, class_name, (ox1, oy1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

    cv2.imshow("show", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
