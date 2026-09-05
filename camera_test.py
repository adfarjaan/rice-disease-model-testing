import cv2

for i in range(5):

    cap = cv2.VideoCapture(i)

    if cap.isOpened():

        ret, frame = cap.read()

        if ret:
            print(f"Camera {i}: FOUND")

            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

        else:
            print(f"Camera {i}: opened but no frame")

        cap.release()

    else:
        print(f"Camera {i}: not found")