import torch
from sam2.build_sam import build_sam2_video_predictor
import supervision as sv
import numpy as np
import cv2

checkpoint = "trained-models/sam2.1_hiera_large.pt"
config = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_model = build_sam2_video_predictor(config, checkpoint, device="cuda")

# Global variables
points = []
labels = []
current_frame = None
frames_list = []


def mouse_callback(event, x, y, flags, param):
    global points, labels, current_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        labels.append(1)
        cv2.circle(current_frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Frame", current_frame)
    elif event == cv2.EVENT_RBUTTONDOWN:
        points.append([x, y])
        labels.append(0)
        cv2.circle(current_frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Frame", current_frame)


# Load video and store frames in memory
video_path = (
    "C:\\Users\\default.DESKTOP-7FKFEEG\\Downloads\\farag elshorbagy 1m chopped.mp4"
)
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames_list.append(frame)
cap.release()

# Initialize SAM2 state with frames in memory
inference_state = sam2_model.init_state(frames_list)

# Create window and set mouse callback
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_callback)

# Start with first frame
frame_idx = 0
current_frame = frames_list[frame_idx]
tracker_id = 1

while True:
    cv2.imshow("Frame", current_frame)
    key = cv2.waitKey(1) & 0xFF

    # Process points
    if key == 32 and len(points) > 0:  # Spacebar
        points_array = np.array(points, dtype=np.float32)
        labels_array = np.array(labels)

        _, object_ids, mask_logits = sam2_model.add_new_points(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=tracker_id,
            points=points_array,
            labels=labels_array,
        )

        # Visualize the mask
        if mask_logits is not None:
            mask = (mask_logits > 0).astype(np.uint8) * 255
            mask_overlay = current_frame.copy()
            mask_overlay[mask > 0] = (
                mask_overlay[mask > 0] * 0.7 + np.array([0, 0, 255]) * 0.3
            )
            cv2.imshow("Mask", mask_overlay)

        points = []
        labels = []

    # Navigate frames
    elif key == ord("n") and frame_idx < len(frames_list) - 1:  # Next frame
        frame_idx += 1
        current_frame = frames_list[frame_idx].copy()
    elif key == ord("p") and frame_idx > 0:  # Previous frame
        frame_idx -= 1
        current_frame = frames_list[frame_idx].copy()
    elif key == ord("q"):  # Quit
        break

cv2.destroyAllWindows()
