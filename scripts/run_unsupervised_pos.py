import argparse, cv2, numpy as np, mediapipe as mp
from scipy.signal import butter, filtfilt, welch

def bandpass(sig, fs, low=0.7, high=3.0, order=3):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, sig)

def pos_projection(rgb_ts):
    X = rgb_ts.T  # 3 x T
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True)+1e-8)
    U = np.array([1, -1, 0], dtype=np.float32)
    V = np.array([1, 1, -2], dtype=np.float32)
    S = U @ X / np.linalg.norm(U) - V @ X / np.linalg.norm(V)
    return S  # 1 x T

def estimate_hr(sig, fs):
    f, Pxx = welch(sig, fs=fs, nperseg=min(len(sig), 256))
    mask = (f >= 0.7) & (f <= 3.0)
    f_sel, P_sel = f[mask], Pxx[mask]
    if len(P_sel) == 0: return None
    hr_hz = f_sel[np.argmax(P_sel)]
    return hr_hz * 60.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to MP4/AVI with a frontal face")
    ap.add_argument("--fps", type=float, default=None, help="Override FPS if missing")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0

    mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    rgb_series = []

    while True:
        ok, frame = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)
        if not res.multi_face_landmarks: continue
        h, w, _ = rgb.shape
        lm = res.multi_face_landmarks[0].landmark

        # Simple polygon over forehead + cheeks
        idxs = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
                379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
                234, 127, 162, 21, 54, 103, 67, 109, 10]
        poly = np.array([(int(lm[i].x*w), int(lm[i].y*h)) for i in idxs], dtype=np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, poly, 255)
        roi = rgb[mask>0]
        if roi.size == 0:
            continue
        rgb_series.append(roi.mean(axis=0))  # mean RGB

    cap.release()

    rgb_ts = np.array(rgb_series)  # T x 3
    if len(rgb_ts) < fps * 5:
        print("Need at least ~5s of video."); return

    sig = pos_projection(rgb_ts).squeeze()
    sig = bandpass(sig, fs=fps)
    hr = estimate_hr(sig, fs=fps)
    if hr is None:
        print("Could not estimate HR.")
    else:
        print(f"Estimated Heart Rate: {hr:.1f} bpm")

if __name__ == "__main__":
    main()
