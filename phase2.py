# ============================================================
# phase2.py — Task 2.1 (Points 1 & 2)
# Run locally with: py -3.11 phase2.py
# Requires: model_v2.keras and label2idx.json in same folder
# ============================================================

import queue, threading, time, json, os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import Counter

# ── Paths ─────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model_v2.h5")
LABEL_PATH = os.path.join(BASE, "label2idx.json")

# ── Constants (must match Phase 1 Cell 3 exactly) ─────────────
FACE_IDX        = [0, 1, 13, 14, 17, 33, 61, 199, 263, 291]
FEATURE_DIM     = 156
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NEUTRAL_IDX     = 4
EMOTION_DIM     = 7
NEUTRAL_EMO     = np.zeros(EMOTION_DIM, dtype=np.float32)
NEUTRAL_EMO[NEUTRAL_IDX] = 1.0
INFERENCE_WIDTH  = 320
INFERENCE_HEIGHT = 240

# ── Queues ─────────────────────────────────────────────────────
frame_queue   = queue.Queue(maxsize=2)
result_queue  = queue.Queue(maxsize=5)
display_queue = queue.Queue(maxsize=2)

# ── Globals ────────────────────────────────────────────────────
_p2_model    = None
_p2_holistic = None
idx2label    = {}


# ── Load model ─────────────────────────────────────────────────
def _load_p2_model():
    global _p2_model, idx2label

    if _p2_model is None:
        print("[inference] Building model and loading weights...")

        # Read num_classes from config
        config_path = os.path.join(BASE, "model_v2_config.json")
        with open(config_path) as f:
            config = json.load(f)
        num_classes = config["num_classes"]

        # Rebuild exact same architecture as Cell 15 in Phase 1
        _p2_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu", input_shape=(163,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ])

        # Build the model by running a dummy prediction
        _p2_model.predict(np.zeros((1, 163), dtype=np.float32), verbose=0)

        # Load weights from numpy file
        weights_path = os.path.join(BASE, "model_v2_weights.npy")
        weights = np.load(weights_path, allow_pickle=True)
        _p2_model.set_weights(list(weights))
        print(f"[inference] Model ready. {num_classes} classes.")

    if not idx2label:
        with open(LABEL_PATH, encoding="utf-8") as f:
            idx2label = {v: k for k, v in json.load(f).items()}
        print(f"[inference] {len(idx2label)} classes loaded.")


# ── Load MediaPipe ─────────────────────────────────────────────
def _load_p2_holistic():
    global _p2_holistic
    if _p2_holistic is None:
        _p2_holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("[inference] MediaPipe Holistic ready (streaming mode).")


# ── Feature extraction (matches Cell 8 exactly) ────────────────
def extract_landmarks_p2(frame_bgr: np.ndarray) -> np.ndarray:
    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = _p2_holistic.process(rgb)
    feat    = []

    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            feat.extend([lm.x, lm.y, lm.z])
    else:
        feat.extend([0.0] * 63)

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            feat.extend([lm.x, lm.y, lm.z])
    else:
        feat.extend([0.0] * 63)

    if results.face_landmarks:
        for idx in FACE_IDX:
            lm = results.face_landmarks.landmark[idx]
            feat.extend([lm.x, lm.y, lm.z])
    else:
        feat.extend([0.0] * 30)

    return np.array(feat, dtype=np.float32)


# ── Normalization (matches Cell 9 exactly) ─────────────────────
def normalize_p2(raw: np.ndarray) -> np.ndarray:
    raw   = raw.astype(np.float64)
    left  = raw[0:63].reshape(21, 3).copy()
    right = raw[63:126].reshape(21, 3).copy()
    face  = raw[126:].reshape(-1, 3).copy()

    if left.any():
        left -= left[0]
        s = np.max(np.linalg.norm(left, axis=1))
        left /= s if s > 0 else 1.0

    if right.any():
        right -= right[0]
        s = np.max(np.linalg.norm(right, axis=1))
        right /= s if s > 0 else 1.0

    if face.any():
        face -= face.mean(axis=0)
        s = np.max(np.linalg.norm(face, axis=1))
        face /= s if s > 0 else 1.0

    return np.concatenate(
        [left.flatten(), right.flatten(), face.flatten()]
    ).astype(np.float32)


# ── Thread 1: Capture ──────────────────────────────────────────
def capture_thread_fn(source, stop_event):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[capture] ERROR: Cannot open '{source}'")
        frame_queue.put(None)
        display_queue.put(None)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[capture] Opened. FPS={fps:.1f}")
    count   = 0
    dropped = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[capture] Stream ended.")
            break

        small = cv2.resize(frame, (INFERENCE_WIDTH, INFERENCE_HEIGHT))

        for q, item in [(frame_queue, small), (display_queue, frame)]:
            try:
                q.put_nowait(item)
            except queue.Full:
                try:    q.get_nowait()
                except queue.Empty: pass
                q.put_nowait(item)
                if q is frame_queue:
                    dropped += 1
        time.sleep(0.05)   # limit to ~20 FPS to match inference speed
        count += 1
        if count % 150 == 0:
            print(f"[capture] frames={count}  dropped={dropped} "
                  f"({100*dropped/count:.1f}%)")

    cap.release()
    frame_queue.put(None)
    display_queue.put(None)
    print(f"[capture] Stopped. total={count}  dropped={dropped}")


# ── Thread 2: Inference ────────────────────────────────────────
def inference_thread_fn(stop_event):
    _load_p2_model()
    _load_p2_holistic()

    frame_times = []

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if frame is None:
            print("[inference] Stop sentinel received. Exiting.")
            break

        t0 = time.perf_counter()

        # Step 1: extract
        raw = extract_landmarks_p2(frame)

        # Step 2: no-hand check
        if raw[:126].sum() == 0.0:
            try:    result_queue.put_nowait(("__no_hands__", 0.0))
            except queue.Full: pass
            continue

        # Step 3: normalize
        norm = normalize_p2(raw)

        # Step 4: concat emotion (Task 2.3 will replace NEUTRAL_EMO)
        feat = np.concatenate([norm, NEUTRAL_EMO]).reshape(1, -1)

        # Step 5: predict
        probs      = _p2_model.predict(feat, verbose=0)[0]
        class_idx  = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        label      = idx2label.get(class_idx, f"cls_{class_idx}")

        # Step 6: push result
        try:    result_queue.put_nowait((label, confidence))
        except queue.Full: pass

        ms = (time.perf_counter() - t0) * 1000
        frame_times.append(ms)

        if len(frame_times) % 50 == 0:
            m = np.mean(frame_times[-50:])
            print(f"[inference] {1000/m:.1f} FPS | {m:.1f}ms | "
                  f"{label} ({confidence:.2f})")

    print("[inference] Thread stopped.")


# ── Display loop (must run in main thread for cv2.imshow) ──────
def run_display(stop_event):
    current_label = "Initializing..."
    current_conf  = 0.0

    while not stop_event.is_set():
        try:
            frame = display_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if frame is None:
            break

        try:
            current_label, current_conf = result_queue.get_nowait()
        except queue.Empty:
            pass

        if current_label == "__no_hands__":
            text, color = "No hand — move closer", (0, 165, 255)
        elif current_label == "Initializing...":
            text, color = "Initializing...", (200, 200, 200)
        else:
            text  = f"Sign: {current_label}  ({current_conf:.2f})"
            color = (0, 200, 0)

        cv2.putText(frame, text, (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2, cv2.LINE_AA)
        cv2.putText(frame, "Emotion: neutral [placeholder]", (12, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow("ESL Recognition — Phase 2", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[main] Q pressed — shutting down.")
            stop_event.set()
            break

    cv2.destroyAllWindows()


# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    VIDEO_SOURCE = 0   # 0 = default webcam

    # Load model and MediaPipe before starting threads
    _load_p2_model()
    _load_p2_holistic()

    stop_event = threading.Event()

    t1 = threading.Thread(
        target=capture_thread_fn,
        args=(VIDEO_SOURCE, stop_event),
        name="Thread-Capture",
        daemon=True,
    )
    t2 = threading.Thread(
        target=inference_thread_fn,
        args=(stop_event,),
        name="Thread-Inference",
        daemon=True,
    )

    print("[main] Starting threads. Press Q in the window to stop.")
    t1.start()
    t2.start()

    # Display runs in main thread — required for cv2.imshow on Windows
    run_display(stop_event)

    # Clean shutdown
    stop_event.set()
    frame_queue.put(None)
    display_queue.put(None)
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    if _p2_holistic:
        _p2_holistic.close()

    print("[main] Shutdown complete.")
    print(f"  Thread-Capture   alive: {t1.is_alive()}")
    print(f"  Thread-Inference alive: {t2.is_alive()}")