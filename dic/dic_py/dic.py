import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# PARAMETERS
# -----------------------------
video_path = r"C:\Users\issac\PyKirigami\dic\dic_video3.mp4"
E_effective = 1.0   # Relative stress

feature_params = dict(
    maxCorners=3000,       # Increase corners to capture lattice
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
# LOAD VIDEO
# -----------------------------
cap = cv2.VideoCapture(video_path)
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read video")

prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
height, width = prev_gray.shape

# Initial texture mask for speckled lattice
lap = cv2.Laplacian(prev_gray, cv2.CV_64F)
texture_mask = np.uint8((np.abs(lap) > np.mean(np.abs(lap))) * 255)

# Detect initial points
p0 = cv2.goodFeaturesToTrack(prev_gray, mask=texture_mask, **feature_params)
if p0 is None:
    raise RuntimeError("No features detected. Check your speckle pattern!")
reference_points = p0.copy()

stress_history = []
frame_count = 0
reseed_interval = 10  # Re-detect features every 10 frames

# -----------------------------
# PROCESS FRAMES
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_count += 1

    # Track points
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)

    # Keep only successfully tracked points
    mask_tracked = st.flatten() == 1
    good_new = p1[mask_tracked][:,0,:]
    good_ref = reference_points[mask_tracked][:,0,:]

    # -----------------------------
    # DISPLACEMENT
    # -----------------------------
    displacement = good_new - good_ref
    ux = displacement[:,0]
    uy = displacement[:,1]

    # -----------------------------
    # AUTOMATIC PISTON FILTER
    # -----------------------------
    disp_mag = np.sqrt(ux**2 + uy**2)
    median_disp = np.median(disp_mag)
    std_disp = np.std(disp_mag)

    # Keep points within median + 2 std
    keep_motion = disp_mag < median_disp + 2*std_disp
    good_new = good_new[keep_motion]
    good_ref = good_ref[keep_motion]
    ux = ux[keep_motion]
    uy = uy[keep_motion]

    # -----------------------------
    # STRAIN CALCULATION
    # -----------------------------
    y_positions = good_ref[:,1]
    if len(y_positions) > 10:
        coeff = np.polyfit(y_positions, uy, 1)
        strain_yy = coeff[0]
        stress = E_effective * strain_yy
        stress_history.append(stress)

    # -----------------------------
    # VISUALIZATION
    # -----------------------------
    vis_frame = frame.copy()
    for new, ref in zip(good_new, good_ref):
        a,b = new
        c,d = ref
        cv2.arrowedLine(vis_frame,
                        (int(c), int(d)),
                        (int(a), int(b)),
                        (0,255,0), 1)

    cv2.imshow("DIC Tracking", vis_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # -----------------------------
    # ITERATIVE RESEEDING OF FEATURES
    # -----------------------------
    if frame_count % reseed_interval == 0:
        # Create mask in lattice areas only (exclude fast points)
        reseed_mask = np.zeros_like(gray, dtype=np.uint8)
        for pt in good_new:
            x,y = int(pt[0]), int(pt[1])
            reseed_mask[y-5:y+5, x-5:x+5] = 255  # small neighborhood

        # Detect new points in lattice
        new_points = cv2.goodFeaturesToTrack(gray, mask=reseed_mask, **feature_params)
        if new_points is not None:
            # Combine old + new points
            good_new_combined = np.vstack([good_new, new_points[:,0,:]])
            good_ref_combined = np.vstack([good_ref, new_points[:,0,:]])
            good_new = good_new_combined
            good_ref = good_ref_combined

    # -----------------------------
    # UPDATE FOR NEXT FRAME
    # -----------------------------
    prev_gray = gray.copy()
    p0 = good_new.reshape(-1,1,2)
    reference_points = good_ref.reshape(-1,1,2)

cap.release()
cv2.destroyAllWindows()

# -----------------------------
# PLOT RELATIVE STRESS
# -----------------------------
plt.plot(stress_history)
plt.xlabel("Frame")
plt.ylabel("Relative Stress (normalized)")
plt.title("Metamaterial Compression Response")
plt.show()