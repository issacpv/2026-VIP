import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# PARAMETERS
# -----------------------------
video_path = r"C:\Users\issac\PyKirigami\dic\video\blue1.mp4"
HSV_FILE = r"C:\Users\issac\PyKirigami\dic\hsv_settings.txt"

TARGET_HEIGHT = 360
E_effective = 1.0

feature_params = dict(
    maxCorners=3000,
    qualityLevel=0.01,
    minDistance=5,
    blockSize=7
)

lk_params = dict(
    winSize=(21,21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# -----------------------------
# RESIZE FUNCTION (360p)
# -----------------------------
def resize_to_360p(frame):
    h, w = frame.shape[:2]
    scale = TARGET_HEIGHT / h
    new_width = int(w * scale)
    return cv2.resize(frame, (new_width, TARGET_HEIGHT),
                      interpolation=cv2.INTER_AREA)

# -----------------------------
# LOAD HSV SETTINGS
# -----------------------------
def load_hsv():
    defaults = [0,0,0,255,255,255]

    if not os.path.exists(HSV_FILE):
        print("HSV file missing — using defaults.")
        return defaults

    try:
        with open(HSV_FILE, "r") as f:
            vals = f.read().strip().split(",")

        if len(vals) != 6:
            return defaults

        return [int(v) for v in vals]
    except:
        return defaults

LH, LS, LV, UH, US, UV = load_hsv()
lower_bound = np.array([LH, LS, LV])
upper_bound = np.array([UH, US, UV])

kernel = np.ones((5,5), np.uint8)

# -----------------------------
# LOAD VIDEO
# -----------------------------
cap = cv2.VideoCapture(video_path)
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read video")

# Resize FIRST frame
first_frame = resize_to_360p(first_frame)

# -----------------------------
# INITIAL HSV MASK
# -----------------------------
hsv0 = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
mask0 = cv2.inRange(hsv0, lower_bound, upper_bound)
mask0 = cv2.morphologyEx(mask0, cv2.MORPH_OPEN, kernel)
mask0 = cv2.morphologyEx(mask0, cv2.MORPH_CLOSE, kernel)

prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.bitwise_and(prev_gray, prev_gray, mask=mask0)

# Texture-based lattice detection
lap = cv2.Laplacian(prev_gray, cv2.CV_64F)
texture_mask = np.uint8((np.abs(lap) > np.mean(np.abs(lap))) * 255)
texture_mask = cv2.bitwise_and(texture_mask, mask0)

p0 = cv2.goodFeaturesToTrack(prev_gray, mask=texture_mask, **feature_params)
if p0 is None:
    raise RuntimeError("No features detected. Check HSV mask!")

reference_points = p0.copy()

stress_history = []
frame_count = 0

# -----------------------------
# PROCESS FRAMES
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = resize_to_360p(frame)
    frame_count += 1

    # HSV MASK
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)

    # -----------------------------
    # OPTICAL FLOW
    # -----------------------------
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, p0, None, **lk_params
    )

    valid = st.flatten() == 1
    good_new = p1[valid][:,0,:]
    good_ref = reference_points[valid][:,0,:]

    displacement = good_new - good_ref
    ux = displacement[:,0]
    uy = displacement[:,1]

    # Remove piston motion (outliers)
    disp_mag = np.sqrt(ux**2 + uy**2)
    median_disp = np.median(disp_mag)
    std_disp = np.std(disp_mag)

    keep = disp_mag < median_disp + 2*std_disp
    good_new = good_new[keep]
    good_ref = good_ref[keep]
    ux = ux[keep]
    uy = uy[keep]

    # -----------------------------
    # STRAIN CALCULATION
    # -----------------------------
    y_positions = good_ref[:,1]
    if len(y_positions) > 10:
        coeff = np.polyfit(y_positions, uy, 1)
        strain_yy = coeff[0]
        stress_history.append(E_effective * strain_yy)

    # -----------------------------
    # VISUALIZATION (HSV OVERLAY)
    # -----------------------------
    overlay = frame.copy()
    overlay[mask > 0] = (0, 255, 255)  # yellow mask highlight

    vis_frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    for new, ref in zip(good_new, good_ref):
        a, b = new
        c, d = ref
        cv2.arrowedLine(
            vis_frame,
            (int(c), int(d)),
            (int(a), int(b)),
            (0,255,0),
            1,
            tipLength=0.3
        )

    cv2.imshow("DIC Tracking (HSV Mask)", vis_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    # UPDATE
    prev_gray = gray.copy()
    p0 = good_new.reshape(-1,1,2)
    reference_points = good_ref.reshape(-1,1,2)

cap.release()
cv2.destroyAllWindows()

# -----------------------------
# PLOT STRESS
# -----------------------------
plt.plot(stress_history)
plt.xlabel("Frame")
plt.ylabel("Relative Stress (normalized)")
plt.title("Metamaterial Compression Response")
plt.show()
