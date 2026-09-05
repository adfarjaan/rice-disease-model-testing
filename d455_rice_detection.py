import cv2
from ultralytics import YOLO


# ============================================================
# 1. LOAD YOLO RICE DISEASE MODEL
# ============================================================

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


# ============================================================
# 2. START LOGITECH WEBCAM
# ============================================================

# Try camera index 0 first
camera_index = 0

cap = cv2.VideoCapture(camera_index)

# Set webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Try 30 FPS
cap.set(cv2.CAP_PROP_FPS, 30)


# Check if camera opened
if not cap.isOpened():

    print("ERROR: Could not open Logitech webcam.")
    print("Try changing camera_index = 0 to camera_index = 1")

    exit()


print("Logitech webcam connected successfully!")
print("Rice Disease Detection started!")
print("Press Q to quit.")


# ============================================================
# 3. MAIN LOOP
# ============================================================

try:

    while True:

        # Get frame from Logitech webcam
        ret, frame = cap.read()

        # Check if frame was received
        if not ret:

            print("ERROR: Could not read frame from webcam.")
            break


        # ====================================================
        # 4. RUN YOLO
        # ====================================================

        results = model(
            frame,
            verbose=False
        )


        # ====================================================
        # 5. DRAW DETECTIONS
        # ====================================================

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])
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

                # Label
                label = (
                    f"{disease_name} "
                    f"{confidence:.2%}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # Print prediction
                print(
                    f"Prediction: {disease_name} "
                    f"| Confidence: {confidence:.2%}"
                )


        # ====================================================
        # 6. DISPLAY LIVE CAMERA
        # ====================================================

        cv2.imshow(
            "Logitech - Rice Disease Detection",
            frame
        )


        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# ============================================================
# 7. ERROR HANDLING
# ============================================================

except Exception as e:

    print("\nERROR:")
    print(e)


# ============================================================
# 8. CLEANUP
# ============================================================

finally:

    cap.release()
    cv2.destroyAllWindows()

    print("\nLogitech webcam stopped.")
    print("Program finished.")