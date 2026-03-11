import cv2
import numpy as np

# ==========================================================
# USER SETTINGS
# ==========================================================
VIDEO_PATH = r"C:\Users\***\VIP-I\poisson\video\blue1.mp4"
HSV_FILE = r"C:\Users\***\VIP-I\hsv\hsvSettings.txt"
OUTPUT_VIDEO = r"C:\Users\***\VIP-I\poisson\video\poisson_output.mp4"

TARGET_HEIGHT = 360
MIN_SPACING_RATIO = 0.10  # 10% spacing for robustness
MAX_MISSING_FRAMES = 2     # max consecutive frames to skip

# ==========================================================
# LOAD HSV SETTINGS
# ==========================================================
with open(HSV_FILE, "r") as f:
    vals = list(map(int, f.read().strip().split(",")))

lower_hsv = np.array(vals[:3])
upper_hsv = np.array(vals[3:])
print("Loaded HSV mask:", lower_hsv, upper_hsv)

# ==========================================================
# VIDEO SETUP
# ==========================================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
scale = TARGET_HEIGHT / orig_h
w = int(orig_w * scale)
h = TARGET_HEIGHT

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (w, h)
)

# ==========================================================
# HELPERS
# ==========================================================
def resize(frame):
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

def mask_material(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# ==========================================================
# INITIAL FRAME
# ==========================================================
ret, frame = cap.read()
frame = resize(frame)
mask = mask_material(frame)
ys, xs = np.where(mask > 0)
if len(xs) == 0:
    raise RuntimeError("Mask failed on initial frame.")

# Initial dimensions
initial_left = xs.min()
initial_right = xs.max()
initial_top = ys.min()
initial_bottom = ys.max()

initial_width = initial_right - initial_left
initial_height = initial_bottom - initial_top
print("Initial width:", initial_width)
print("Initial height:", initial_height)

# ==========================================================
# TRACK MAX COMPRESSION
# ==========================================================
min_height = initial_height
width_at_min_height = initial_width

height_buffer = []
width_buffer = []
frame_counter = 0

# ==========================================================
# PROCESS VIDEO
# ==========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = resize(frame)
    mask = mask_material(frame)
    ys, xs = np.where(mask > 0)

    display = frame.copy()

    if len(xs) == 0 or len(ys) == 0:
        # Skip frame if mask fails
        cv2.imshow("Specimen Tracking", display)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        continue

    # Robust height
    top = ys.min()
    bottom = ys.max()
    height = bottom - top

    # Robust width using percentiles
    left = np.percentile(xs, 5)
    right = np.percentile(xs, 95)
    width = right - left

    # Draw bounding box
    cv2.rectangle(display, (int(left), top), (int(right), bottom), (0,0,255), 2)

    # Buffer measurements for failsafe (check neighboring frames)
    height_buffer.append(height)
    width_buffer.append(width)
    frame_counter += 1

    if frame_counter >= 3:
        prev_h, curr_h, next_h = height_buffer[-3:]
        prev_w, curr_w, next_w = width_buffer[-3:]
        if curr_h <= prev_h and curr_h <= next_h:
            if curr_h < min_height:
                min_height = curr_h
                width_at_min_height = curr_w

    cv2.imshow("Specimen Tracking", display)
    writer.write(display)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# ==========================================================
# FINAL STRAIN & POISSON
# ==========================================================
final_height = min_height
final_width = width_at_min_height

eps_axial = (final_height - initial_height) / initial_height
eps_lateral = (final_width - initial_width) / initial_width
poisson_ratio = -eps_lateral / eps_axial

# ==========================================================
# PRINT RESULTS
# ==========================================================
print("\n==============================")
print("Initial width:", initial_width)
print("Initial height:", initial_height)
print("Final width:", final_width)
print("Final height:", final_height)
print("\nAxial strain:", eps_axial)
print("Lateral strain:", eps_lateral)
print("Poisson ratio:", poisson_ratio)

# ==========================================================
# CLEANUP
# ==========================================================
cap.release()
writer.release()
cv2.destroyAllWindows()
print("\nFinished. Output saved to:", OUTPUT_VIDEO)
