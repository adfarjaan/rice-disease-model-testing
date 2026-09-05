import cv2
from ultralytics import YOLO


# ==========================================
# 1. LOAD YOUR RICE DISEASE MODEL
# ==========================================

model = YOLO("YOLO11L-RICE-Disease-Detection.pt")

class_names = {
    0: "Bacterial Leaf Blight",
    1: "Brown Spot",
    2: "Healthy Leaf",
    3: "Leaf Blast",
    4: "Leaf Scald",
    5: "Narrow Brown Leaf Spot",
    6: "Neck Blast",
    7: "Rice Hispa"
}


# ==========================================
# 2. START LOGITECH WEBCAM
# ==========================================

camera_index = 3

cap = cv2.VideoCapture(camera_index)

# Request 640 x 480 at 30 FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():

    print("ERROR: Could not open Logitech webcam.")
    print("Try changing camera_index = 0 to camera_index = 1")

    exit()


print("======================================")
print("Logitech webcam started")
print("Rice Disease Detection started")
print("Press Q to quit")
print("======================================")


# ==========================================
# 3. READ LIVE CAMERA FRAMES
# ==========================================

try:

    while True:

        # Get frame from Logitech webcam
        ret, frame = cap.read()

        if not ret:

            print("ERROR: Could not read webcam frame.")
            break


        # ==================================
        # 4. RUN YOUR YOLO MODEL
        # ==================================

        results = model(
            frame,
            verbose=False
        )


        # ==================================
        # 5. DISPLAY DETECTIONS
        # ==================================

        for result in results:

            for box in result.boxes:

                # Class ID
                class_id = int(box.cls[0])

                # Confidence
                confidence = float(box.conf[0])

                # Bounding box coordinates
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # Get disease name
                disease_name = class_names.get(
                    class_id,
                    "Unknown"
                )

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Text
                label = f"{disease_name} {confidence:.2%}"

                # Draw label
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # Print prediction in PowerShell
                print(
                    f"Prediction: {disease_name} "
                    f"| Confidence: {confidence:.2%}"
                )


        # ==================================
        # 6. SHOW LIVE RESULT
        # ==================================

        cv2.imshow(
            "Logitech - Rice Disease Detection",
            frame
        )


        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


finally:

    # ==========================================
    # 7. STOP WEBCAM
    # ==========================================

    cap.release()
    cv2.destroyAllWindows()

    print("Logitech webcam stopped.")
    print("Program finished.")