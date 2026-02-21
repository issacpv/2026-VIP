import cv2
import numpy as np
import os

# ================= SETTINGS =================
TESTING_MODE = 0  # 0 = save/load HSV, 1 = preset mode (no saving)

HSV_FILE = r"C:\Users\issac\PyKirigami\dic\hsv_settings.txt"
TARGET_HEIGHT = 360
# ============================================

def nothing(x):
    pass


# ---------- Load HSV settings ----------
def load_hsv_defaults():
    # Default values
    defaults = [0, 0, 0, 255, 255, 255]

    if TESTING_MODE == 1:
        return defaults

    if not os.path.exists(HSV_FILE):
        return defaults

    try:
        with open(HSV_FILE, "r") as f:
            values = f.read().strip().split(",")

        if len(values) != 6:
            return defaults

        return [int(v) for v in values]

    except:
        return defaults


# ---------- Save HSV settings ----------
def save_hsv(values):
    if TESTING_MODE == 1:
        return

    try:
        with open(HSV_FILE, "w") as f:
            f.write(",".join(map(str, values)))
        print("HSV settings saved.")
    except Exception as e:
        print("Failed to save HSV:", e)


# ---------- Initialize ----------
hsv_defaults = load_hsv_defaults()

cap = cv2.VideoCapture(
    r"C:\Users\issac\PyKirigami\dic\video\blue1Long.mp4"
)

cv2.namedWindow("Tracking")

cv2.createTrackbar("LH", "Tracking", hsv_defaults[0], 255, nothing)
cv2.createTrackbar("LS", "Tracking", hsv_defaults[1], 255, nothing)
cv2.createTrackbar("LV", "Tracking", hsv_defaults[2], 255, nothing)
cv2.createTrackbar("UH", "Tracking", hsv_defaults[3], 255, nothing)
cv2.createTrackbar("US", "Tracking", hsv_defaults[4], 255, nothing)
cv2.createTrackbar("UV", "Tracking", hsv_defaults[5], 255, nothing)

paused = False
frame = None

kernel = np.ones((5,5), np.uint8)

# ---------- Main Loop ----------
while True:

    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to 360p
        h, w = frame.shape[:2]
        scale = TARGET_HEIGHT / h
        new_width = int(w * scale)

        frame = cv2.resize(
            frame,
            (new_width, TARGET_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

    if frame is None:
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Read sliders
    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")
    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # ---- Speckle removal ----
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)

    key = cv2.waitKey(30) & 0xFF

    # -------- Quit --------
    if key == ord('q') or key == 27:
        save_hsv([l_h, l_s, l_v, u_h, u_s, u_v])
        break

    # -------- Pause --------
    elif key == 32:
        paused = not paused

# ---------- Cleanup ----------
cap.release()
cv2.destroyAllWindows()