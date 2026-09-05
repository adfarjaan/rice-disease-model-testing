import pyrealsense2 as rs
import cv2

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

print("D455 connected! Press Q to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        color_image = cv2.cvtColor(
            __import__("numpy").asanyarray(color_frame.get_data()),
            cv2.COLOR_RGB2BGR
        )

        cv2.imshow("D455 RGB Camera", color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()