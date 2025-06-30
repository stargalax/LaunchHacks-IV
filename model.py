import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict

model = YOLO('yolov8n.pt')
tracker = DeepSort(max_age=15)
GRID_SIZE = 100
ALERT_THRESHOLD = 3

def process_video_stream(path, sector_id):
    cap = cv2.VideoCapture(path)
    trajectories = defaultdict(list)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        people_positions = []
        results = model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if model.names[cls_id] == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

        print(f"[{sector_id}] Processing frame, {len(detections)} people detected")

        if len(detections) == 0:
            yield frame, False
            continue

        try:
            tracks = tracker.update_tracks(detections, frame=frame)
        except Exception as e:
            print(f"[{sector_id}] DeepSORT error: {e}")
            yield frame, False
            continue

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            people_positions.append((cx, cy))
            trajectories[track_id].append((cx, cy))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            for i in range(1, len(trajectories[track_id])):
                cv2.line(frame, trajectories[track_id][i - 1],
                         trajectories[track_id][i], (255, 0, 0), 2)

        frame_h, frame_w = frame.shape[:2]
        grid_rows = frame_h // GRID_SIZE
        grid_cols = frame_w // GRID_SIZE
        density_grid = np.zeros((grid_rows, grid_cols), dtype=int)

        for x, y in people_positions:
            col = x // GRID_SIZE
            row = y // GRID_SIZE
            if 0 <= row < grid_rows and 0 <= col < grid_cols:
                density_grid[row, col] += 1

        alert_triggered = False
        for r in range(grid_rows):
            for c in range(grid_cols):
                x1, y1 = c * GRID_SIZE, r * GRID_SIZE
                x2, y2 = x1 + GRID_SIZE, y1 + GRID_SIZE
                count = density_grid[r, c]
                if count >= ALERT_THRESHOLD:
                    color = (0, 0, 255)
                    alert_triggered = True
                elif count == 2:
                    color = (0, 165, 255)
                elif count == 1:
                    color = (0, 255, 0)
                else:
                    color = (100, 100, 100)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                if count > 0:
                    cv2.putText(frame, str(count), (x1 + 3, y1 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if alert_triggered:
            cv2.putText(frame, f"ALERT: {sector_id} overcrowded!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        yield frame, alert_triggered

    cap.release()
