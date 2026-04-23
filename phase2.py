# ============================================================
# PHASE 2: Live Inference System (Sign Language & Emotion)
# ============================================================
# This script implements a multi-threaded system to:
# 1. Capture real-time video frames.
# 2. Extract Landmarks using MediaPipe.
# 3. Detect facial emotions using DeepFace.
# 4. Predict Egyptian Sign Language using a trained Keras model.
# ============================================================

import queue, threading, time, json, os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from deepface import DeepFace
from collections import deque, Counter

# Hide TensorFlow unnecessary logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Global Settings & Shared States ──────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
LABEL_PATH = os.path.join(BASE, "label2idx.json")
NO_FACE_SENTINEL = "No face detected"

# Thread safety locks
emotion_lock = threading.Lock()
dominant_emotion = "neutral"
dominant_emotion_score = 0.0
emotion_history = deque(maxlen=5) 

face_lock = threading.Lock()
face_detected = False

# ── Constants ────────────────────────────────────────────────
FACE_IDX = [0, 1, 13, 14, 17, 33, 61, 199, 263, 291]
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
INFERENCE_WIDTH, INFERENCE_HEIGHT = 320, 240
EMOTION_ANALYSIS_FREQ = 3 

# ── Communication Queues ─────────────────────────────────────
frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=5)
display_queue = queue.Queue(maxsize=2)
emotion_frame_queue = queue.Queue(maxsize=2)

# ── Globals ──────────────────────────────────────────────────
_p2_model = None
_p2_holistic = None
idx2label = {}

# ── Utility: Normalize Landmarks (Matches Training Cell 9) ──
def normalize_frame(raw):
    """Normalize a 156-dim raw feature vector for the model."""
    raw = raw.astype(np.float64)
    left = raw[0:63].reshape(21, 3).copy()
    right = raw[63:126].reshape(21, 3).copy()
    face = raw[126:].reshape(-1, 3).copy()

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

    return np.concatenate([left.flatten(), right.flatten(), face.flatten()]).astype(np.float32)

# ── Helper: Load Resources ──────────────────────────────────
def _load_p2_model():
    global _p2_model, idx2label
    if _p2_model is None:
        print("[inference] Building model and loading weights...")
        config_path = os.path.join(BASE, "model_v2_config.json")
        with open(config_path) as f:
            config = json.load(f)
        num_classes = config["num_classes"]

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
        _p2_model.predict(np.zeros((1, 163), dtype=np.float32), verbose=0)
        weights_path = os.path.join(BASE, "model_v2_weights.npy")
        _p2_model.set_weights(list(np.load(weights_path, allow_pickle=True)))
        
        with open(LABEL_PATH, encoding="utf-8") as f:
            idx2label = {v: k for k, v in json.load(f).items()}

def _load_p2_holistic():
    global _p2_holistic
    if _p2_holistic is None:
        _p2_holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )

# ============================================================
# THREAD 1: EMOTION ANALYSIS (DeepFace)
# ============================================================
# Task: Periodically analyze facial emotions to provide context.
def emotion_thread_fn(stop_event):
    global dominant_emotion, dominant_emotion_score
    while not stop_event.is_set():
        try:
            frame = emotion_frame_queue.get(timeout=1.0)
            if frame is None: break
            
            with face_lock: 
                active = face_detected
            
            if not active:
                with emotion_lock:
                    dominant_emotion = NO_FACE_SENTINEL
                    dominant_emotion_score = 0.0
                    emotion_history.clear()
                continue
            try:
                analysis = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=True, silent=True)
                if analysis:
                    emo_raw = analysis[0]["dominant_emotion"]
                    score_raw = analysis[0]["emotion"][emo_raw]
                    emotion_history.append(emo_raw)
                    most_common_emo = Counter(emotion_history).most_common(1)[0][0]
                    with emotion_lock:
                        dominant_emotion = most_common_emo
                        dominant_emotion_score = score_raw
            except:
                with emotion_lock: 
                    dominant_emotion = NO_FACE_SENTINEL
        except: 
            continue

# ============================================================
# THREAD 2: CAMERA CAPTURE
# ============================================================
# Task: Constantly read frames from webcam and distribute to other threads.
def capture_thread_fn(source, stop_event):
    cap = cv2.VideoCapture(source)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: break
        
        small = cv2.resize(frame, (INFERENCE_WIDTH, INFERENCE_HEIGHT))
        # Feed all processing queues
        for q, item in [(frame_queue, small), (display_queue, frame), (emotion_frame_queue, frame)]:
            try: 
                q.put_nowait(item)
            except:
                try: 
                    q.get_nowait()
                    q.put_nowait(item)
                except: pass
        time.sleep(0.04)
    cap.release()

# ============================================================
# THREAD 3: SIGN INFERENCE (MediaPipe + Keras)
# ============================================================
# Task: Extract landmarks, normalize them, and predict the sign class.
def inference_thread_fn(stop_event):
    _load_p2_model()
    _load_p2_holistic()
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1.0)
        except: 
            continue
        
        results = _p2_holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        feat = []
        
        # 1. Process Hands
        for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand: 
                [feat.extend([lm.x, lm.y, lm.z]) for lm in hand.landmark]
            else: 
                feat.extend([0.0] * 63)
            
        # 2. Process Face & Sync with Emotion Thread
        if results.face_landmarks:
            for idx in FACE_IDX:
                lm = results.face_landmarks.landmark[idx]
                feat.extend([lm.x, lm.y, lm.z])
            with face_lock: 
                global face_detected
                face_detected = True
        else:
            feat.extend([0.0] * 30)
            with face_lock: 
                face_detected = False

        raw = np.array(feat, dtype=np.float32)
        
        # Check if at least one hand is detected
        if raw[:126].sum() == 0.0:
            result_queue.put(("__no_hands__", 0.0))
            continue
            
        # ── NORMALIZATION ──
        norm = normalize_frame(raw)
        
        with emotion_lock:
            emo_label = dominant_emotion
            emo_score = dominant_emotion_score
        
        # Prepare emotion one-hot vector (7-dim)
        emo_vec = np.zeros(7, dtype=np.float32)
        if emo_label in EMOTION_CLASSES:
            emo_vec[EMOTION_CLASSES.index(emo_label)] = 1.0
        else: 
            emo_vec[4] = 1.0 # Default: Neutral
        
        # Final prediction input: 156 landmarks + 7 emotion = 163 total
        feat_final = np.concatenate([norm, emo_vec]).reshape(1, -1)
        probs = _p2_model.predict(feat_final, verbose=0)[0]
        idx = int(np.argmax(probs))
        res_label = idx2label.get(idx, "unknown")
        sign_conf = float(probs[idx])
        
        # Console output for debugging
        print(f"[System] Sign: {res_label:12} ({sign_conf:.2f}) | Emotion: {emo_label:8} ({emo_score:.1f}%)")
        result_queue.put((res_label, sign_conf))

# ============================================================
# THREAD 4: MAIN DISPLAY (GUI)
# ============================================================
# Task: Read results and display the final UI to the user.
if __name__ == "__main__":
    _load_p2_model()
    _load_p2_holistic()
    stop_event = threading.Event()
    
    # Initialize Background Threads
    t_cap = threading.Thread(target=capture_thread_fn, args=(0, stop_event), daemon=True)
    t_inf = threading.Thread(target=inference_thread_fn, args=(stop_event,), daemon=True)
    t_emo = threading.Thread(target=emotion_thread_fn, args=(stop_event,), daemon=True)
    
    t_cap.start()
    t_inf.start()
    t_emo.start()
    
    curr_label, curr_conf = "Initializing...", 0.0
    
    # Main GUI Loop (Must run in Main Thread)
    while not stop_event.is_set():
        try:
            frame = display_queue.get(timeout=1.0)
            try: 
                curr_label, curr_conf = result_queue.get_nowait()
            except: pass
            
            with emotion_lock: 
                emo = dominant_emotion

            # Determine UI Text and Color
            if curr_label == "__no_hands__":
                msg, color = "No hands detected", (0, 165, 255)
            elif not face_detected:
                msg, color = "No face detected", (0, 0, 255)
            else:
                msg, color = f"Sign: {curr_label} ({curr_conf:.2f})", (0, 255, 0)

            # Draw Overlays
            cv2.putText(frame, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Emotion: {emo}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("ESL Final System - Phase 2", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                stop_event.set()
                break
        except: 
            continue

    cv2.destroyAllWindows()