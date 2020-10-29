import argparse
import cv2
from imutils.video import VideoStream, FPS
import time
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--target", type=str,
                help="Dataset target file path")
ap.add_argument("-v", "--video", type=str,
                help="Path to input video file")
ap.add_argument("-o", "--option", type=str, default="kcf",
                help="OpenCV object tracker type")
args = vars(ap.parse_args())

major, minor = cv2.__version__.split(".")[:2]

if int(major) == 3 and int(minor) < 3:
    tracker = cv2.Tracker_ceate(args["tracker"].uppper())
else:
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }
    tracker = OPENCV_OBJECT_TRACKERS[args["option"]]()

initBB = None

if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    # vs = cv2.VideoCapture(0)
    time.sleep(1.0)
else:
    vs = cv2.VideoCapture(args["video"])

fps = None
while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    if frame is None:
        break
    frame = imutils.resize(frame, width=500)
    H, W = frame.shape[:2]

    if initBB is not None:
        success, box = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            roi = cv2.cvtColor(frame[y + 1:(y + h) - 1, x + 1:(x + w) - 1], cv2.COLOR_BGR2GRAY)
            cv2.imwrite(args["target"] + "/{}.jpg".format(time.time()), roi)

        fps.update()
        fps.stop()

        info = [
            ("Tracker", args["option"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps()))
        ]

        for (i, (key, value)) in enumerate(info):
            text = "{}: {}".format(key, value)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Tracking", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        initBB = cv2.selectROI("Tracking", frame, fromCenter=False, showCrosshair=True)
        tracker = OPENCV_OBJECT_TRACKERS[args["option"]]()
        tracker.init(frame, initBB)
        fps = FPS().start()

    elif key == ord("q"):
        break
if not args.get("video", False):
    vs.stop()
else:
    vs.release()
cv2.destroyAllWindows()
