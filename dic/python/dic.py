import cv2
import numpy as np

# ==========================================================
# USER SETTINGS
# ==========================================================
VIDEO_PATH = r"C:\Users\issac\PyKirigami\dic\video\blue1.mp4"
HSV_FILE = r"C:\Users\issac\PyKirigami\dic\hsvSettings.txt"
OUTPUT_VIDEO = r"C:\Users\issac\PyKirigami\dic\dic_output.mp4"

TARGET_HEIGHT = 360   # <-- force 360p processing

ARROW_SPACING = 12
FLOW_SCALE = 4
HEAT_ALPHA = 0.55
GAUSSIAN_BLUR = 5

# ==========================================================
# LOAD HSV SETTINGS
# ==========================================================
with open(HSV_FILE, "r") as f:
    vals = list(map(int, f.read().strip().split(",")))

hmin, smin, vmin, hmax, smax, vmax = vals
lower_hsv = np.array([hmin, smin, vmin])
upper_hsv = np.array([hmax, smax, vmax])

print("Loaded HSV mask:", lower_hsv, upper_hsv)

# ==========================================================
# VIDEO LOAD
# ==========================================================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ---- Compute 360p resize ----
scale = TARGET_HEIGHT / orig_h
w = int(orig_w * scale)
h = TARGET_HEIGHT

print(f"Resizing video from {orig_w}x{orig_h} → {w}x{h}")

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (w, h)
)

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def resize_frame(frame):
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def get_material_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def compute_strain(flow):
    fx = flow[...,0]
    fy = flow[...,1]

    dfx_dx = cv2.Sobel(fx, cv2.CV_32F, 1, 0, ksize=3)
    dfx_dy = cv2.Sobel(fx, cv2.CV_32F, 0, 1, ksize=3)
    dfy_dx = cv2.Sobel(fy, cv2.CV_32F, 1, 0, ksize=3)
    dfy_dy = cv2.Sobel(fy, cv2.CV_32F, 0, 1, ksize=3)

    strain = np.sqrt(
        dfx_dx**2 +
        dfy_dy**2 +
        0.5*(dfx_dy + dfy_dx)**2
    )

    return strain


# ==========================================================
# INITIAL FRAME
# ==========================================================
ret, prev_frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read first frame")

prev_frame = resize_frame(prev_frame)

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_mask = get_material_mask(prev_frame)
prev_gray = cv2.bitwise_and(prev_gray, prev_gray, mask=prev_mask)

paused = False

# ==========================================================
# MAIN LOOP
# ==========================================================
while True:

    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        # ---- RESIZE BEFORE PROCESSING ----
        frame = resize_frame(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = get_material_mask(frame)
        gray = cv2.bitwise_and(gray, gray, mask=mask)

        # Dense Optical Flow (DIC)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            0.5, 5, 35, 5, 7, 1.5, 0
        )

        # -------- Heatmap --------
        strain = compute_strain(flow)
        strain = cv2.GaussianBlur(strain, (GAUSSIAN_BLUR, GAUSSIAN_BLUR), 0)

        norm = cv2.normalize(strain, None, 0, 255, cv2.NORM_MINMAX)
        norm = norm.astype(np.uint8)

        heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        heatmap = cv2.bitwise_and(heatmap, heatmap, mask=mask)

        overlay = cv2.addWeighted(frame, 1-HEAT_ALPHA, heatmap, HEAT_ALPHA, 0)

        # -------- Draw Arrows --------
        for y in range(0, h, ARROW_SPACING):
            for x in range(0, w, ARROW_SPACING):

                if mask[y, x] == 0:
                    continue

                dx, dy = flow[y, x]
                end = (int(x + dx*FLOW_SCALE), int(y + dy*FLOW_SCALE))

                cv2.arrowedLine(
                    overlay, (x, y), end,
                    (255,255,255), 1, tipLength=0.3
                )

        writer.write(overlay)
        cv2.imshow("DIC Stress Mapping (360p)", overlay)

        prev_gray = gray.copy()

    # -------- Keyboard Controls --------
    key = cv2.waitKey(30) & 0xFF

    if key == 32:   # SPACE
        paused = not paused
        print("Paused" if paused else "Resumed")

    elif key == ord('q'):
        print("Force quitting...")
        break

# ==========================================================
# CLEANUP
# ==========================================================
cap.release()
writer.release()
cv2.destroyAllWindows()

print("Finished. Output saved to:", OUTPUT_VIDEO)
