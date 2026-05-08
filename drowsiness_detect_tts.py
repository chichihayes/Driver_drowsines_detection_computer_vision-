import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
import numpy as np
import threading
import time
import collections
import urllib.request
import os
import csv
import json
from scipy.spatial import distance
from openai import OpenAI
import pygame
import io

# ─── OpenAI TTS Setup ────────────────────────────────────────────────────────
OPENAI_API_KEY = ""
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ─── MediaPipe Model ──────────────────────────────────────────────────────────
MODEL_PATH = "face_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading face landmark model (~30MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

# ─── Alert Messages ───────────────────────────────────────────────────────────
ALERT_MESSAGES = {
    "warning_eye": [
        "Your eyes are closing. Open them up and keep them on the road.",
        "Eyes dropping detected. Wake up and stay focused.",
    ],
    "warning_head": [
        "Your head is dropping. Sit up straight and stay in control.",
        "Head movement detected. Lift your head and keep your eyes forward.",
    ],
    "warning_combined": [
        "Your eyes and head are both dropping. Pull over soon before it gets worse.",
        "Eyes closing and head dropping at the same time. Find a safe spot to stop.",
    ],
    "critical": [
        "You are falling asleep. Pull over right now, this is an emergency.",
        "Severe drowsiness detected. Stop the vehicle safely and immediately.",
    ],
    "recovery": [
        "Eyes open, head up. Stay sharp and drive safe.",
        "Good, you are alert again. Keep your focus on the road.",
    ]
}

# ─── OpenAI TTS Alert Engine ─────────────────────────────────────────────────
AUDIO_CACHE_DIR = "alert_audio_cache"

def pregenerate_audio():
    """Generate all alert MP3s once at startup and save to disk."""
    os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)
    cache = {}
    total = sum(len(v) for v in ALERT_MESSAGES.values())
    done  = 0
    print(f"Pre-generating {total} audio alerts...")
    for level, messages in ALERT_MESSAGES.items():
        cache[level] = []
        for i, text in enumerate(messages):
            path = os.path.join(AUDIO_CACHE_DIR, f"{level}_{i}.mp3")
            if not os.path.exists(path):
                try:
                    response = openai_client.audio.speech.create(
                        model="tts-1",
                        voice="nova",
                        input=text,
                        speed=1.1,
                    )
                    with open(path, "wb") as f:
                        f.write(response.content)
                except Exception as e:
                    print(f"[TTS Pregen Error] {level}_{i}: {e}")
                    path = None
            done += 1
            print(f"  [{done}/{total}] {level}_{i} ready")
            cache[level].append(path)
    print("All audio ready. Starting detection...\n")
    return cache


class OpenAIAlertEngine:
    def __init__(self, cache):
        pygame.mixer.init()
        self.speaking    = False
        self.alert_index = {k: 0 for k in ALERT_MESSAGES}
        self.cache       = cache

    def speak(self, level):
        if self.speaking:
            return
        paths = self.cache.get(level, [])
        if not paths:
            return
        idx  = self.alert_index[level] % len(paths)
        path = paths[idx]
        self.alert_index[level] += 1
        if not path or not os.path.exists(path):
            return

        def run():
            self.speaking = True
            try:
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
            except Exception as e:
                print(f"[Audio Error] {e}")
            finally:
                self.speaking = False

        threading.Thread(target=run, daemon=True).start()

    def stop(self):
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except:
            pass


# ─── Eye / Head Landmark Indices ──────────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ─── Detection Thresholds ─────────────────────────────────────────────────────
EAR_THRESHOLD     = 0.22
HEAD_PITCH_THRESH = 15.0
WARNING_FRAMES    = 20
CRITICAL_FRAMES   = 40
EAR_HISTORY_LEN   = 10
HEAD_HISTORY_LEN  = 10


def compute_ear(landmarks, eye_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.3


def compute_head_pitch(landmarks, w, h):
    nose   = landmarks[1]
    chin   = landmarks[152]
    brow   = landmarks[10]
    face_h = (chin.y - brow.y) * h
    if face_h < 1:
        return 0.0
    pitch = np.degrees(np.arctan2(chin.z - nose.z, face_h / w))
    return pitch


def draw_eye_outline(frame, landmarks, eye_indices, w, h, color):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    for i in range(len(pts)):
        cv2.line(frame, pts[i], pts[(i + 1) % len(pts)], color, 1)


def draw_hud(frame, ear, pitch, state, blink_count, drowsy_events, elapsed, eye_warn, head_warn):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    if state == "ALERT":
        bar_color    = (0, 200, 0)
        status_color = (0, 255, 0)
        status_text  = "ALERT"
    elif state == "CRITICAL":
        bar_color    = (0, 0, 220)
        status_color = (0, 0, 255)
        status_text  = "CRITICAL"
    else:
        bar_color    = (0, 165, 255)
        status_color = (0, 165, 255)
        if state == "WARNING_EYE":
            status_text = "WARNING — EYES CLOSED"
        elif state == "WARNING_HEAD":
            status_text = "WARNING — HEAD DROP"
        elif state == "WARNING_COMBINED":
            status_text = "WARNING — EYES + HEAD"
        else:
            status_text = "WARNING"

    cv2.rectangle(overlay, (0, 0), (w, 130), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    eye_col  = (0, 0, 255) if eye_warn  else (0, 255, 0)
    head_col = (0, 0, 255) if head_warn else (0, 255, 0)

    cv2.putText(frame, f"EAR: {ear:.3f}",       (10, 28),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, eye_col,      2)
    cv2.putText(frame, f"HEAD: {pitch:.1f}deg",  (10, 55),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, head_col,     2)
    cv2.putText(frame, f"Status: {status_text}", (10, 82),  cv2.FONT_HERSHEY_SIMPLEX, 0.70, status_color, 2)
    cv2.putText(frame, f"Blinks: {blink_count}   Drowsy Events: {drowsy_events}",
                (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    mins = int(elapsed) // 60
    secs = int(elapsed) % 60
    cv2.putText(frame, f"{mins:02d}:{secs:02d}", (w - 80, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    cv2.putText(frame, "[EYE]",  (w - 130, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_col,  2)
    cv2.putText(frame, "[HEAD]", (w - 130, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, head_col, 2)

    bar_w = max(0, min(int((ear / 0.4) * (w - 20)), w - 20))
    cv2.rectangle(frame, (10, h - 20), (w - 10, h - 8), (60, 60, 60), -1)
    cv2.rectangle(frame, (10, h - 20), (10 + bar_w, h - 8), bar_color, -1)
    cv2.putText(frame, "EAR", (10, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    if state == "CRITICAL":
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 200), 5)
    elif state in ("WARNING_EYE", "WARNING_HEAD", "WARNING_COMBINED"):
        cv2.rectangle(frame, (0, 0), (w, h), (0, 120, 255), 3)


# ─── Session Logger ───────────────────────────────────────────────────────────
class SessionLogger:
    """
    Writes two files per session:
      session_YYYYMMDD_HHMMSS/
        frame_log.csv     — one row per frame: timestamp, ear, pitch, state, ...
        event_log.csv     — one row per state transition: timestamp, from_state, to_state, latency_ms
        session_summary.json — end-of-session aggregated metrics
        screenshots/      — HUD frame saved on every state change
    """

    def __init__(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = f"session_{ts}"
        self.screenshot_dir = os.path.join(self.session_dir, "screenshots")
        os.makedirs(self.screenshot_dir, exist_ok=True)

        # Frame-level CSV
        self.frame_csv_path = os.path.join(self.session_dir, "frame_log.csv")
        self._frame_file = open(self.frame_csv_path, "w", newline="")
        self._frame_writer = csv.writer(self._frame_file)
        self._frame_writer.writerow([
            "timestamp_s", "ear_raw", "ear_smooth", "pitch_raw", "pitch_smooth",
            "state", "closed_frames", "head_frames", "eye_warn", "head_warn"
        ])

        # Event-level CSV (state transitions)
        self.event_csv_path = os.path.join(self.session_dir, "event_log.csv")
        self._event_file = open(self.event_csv_path, "w", newline="")
        self._event_writer = csv.writer(self._event_file)
        self._event_writer.writerow([
            "timestamp_s", "from_state", "to_state",
            "ear_at_transition", "pitch_at_transition",
            "frames_in_prev_state", "latency_from_onset_ms"
        ])

        # Internal tracking
        self._prev_state          = "ALERT"
        self._state_entry_time    = time.time()
        self._state_entry_elapsed = 0.0
        self._onset_time          = None   # when eye/head first went bad
        self._onset_elapsed       = None
        self._frame_count_in_state = 0

        # State time accumulators (seconds)
        self.state_durations = {
            "ALERT": 0.0,
            "WARNING_EYE": 0.0,
            "WARNING_HEAD": 0.0,
            "WARNING_COMBINED": 0.0,
            "CRITICAL": 0.0,
        }

        # Per-state EAR lists for summary stats
        self._ear_by_state = {k: [] for k in self.state_durations}

        self._screenshot_count = 0

        print(f"[Logger] Session dir: {self.session_dir}")

    def log_frame(self, elapsed, ear_raw, ear_smooth, pitch_raw, pitch_smooth,
                  state, closed_frames, head_frames, eye_warn, head_warn, frame):
        """Call once per frame."""
        self._frame_writer.writerow([
            f"{elapsed:.3f}", f"{ear_raw:.4f}", f"{ear_smooth:.4f}",
            f"{pitch_raw:.2f}", f"{pitch_smooth:.2f}",
            state, closed_frames, head_frames,
            int(eye_warn), int(head_warn)
        ])

        # Track EAR per state
        self._ear_by_state[state].append(ear_smooth)

        # Accumulate state duration
        now = time.time()
        self.state_durations[state] += (now - self._state_entry_time) if state == self._prev_state else 0

        # Track onset (moment eye_warn or head_warn first became True this episode)
        if (eye_warn or head_warn) and self._onset_time is None:
            self._onset_time    = now
            self._onset_elapsed = elapsed

        if not eye_warn and not head_warn:
            self._onset_time    = None
            self._onset_elapsed = None

        # State transition
        if state != self._prev_state:
            # Calculate latency from onset to this state change
            latency_ms = round((now - self._onset_time) * 1000, 1) if self._onset_time else 0

            self._event_writer.writerow([
                f"{elapsed:.3f}",
                self._prev_state,
                state,
                f"{ear_smooth:.4f}",
                f"{pitch_smooth:.2f}",
                self._frame_count_in_state,
                latency_ms
            ])
            self._event_file.flush()

            # Screenshot on every state change
            screenshot_name = (
                f"{self._screenshot_count:03d}_"
                f"{self._prev_state}_to_{state}_"
                f"{elapsed:.1f}s.png"
            )
            cv2.imwrite(
                os.path.join(self.screenshot_dir, screenshot_name),
                frame
            )
            self._screenshot_count += 1

            # Accumulate duration for the state we just left
            duration_in_prev = now - self._state_entry_time
            self.state_durations[self._prev_state] += duration_in_prev

            self._prev_state           = state
            self._state_entry_time     = now
            self._state_entry_elapsed  = elapsed
            self._frame_count_in_state = 0
        else:
            self._frame_count_in_state += 1

        self._frame_file.flush()

    def save_summary(self, elapsed, blink_count, drowsy_events, fps_actual):
        """Call at session end."""
        # Finalize duration for last state
        self.state_durations[self._prev_state] += time.time() - self._state_entry_time

        # Compute EAR stats per state
        ear_stats = {}
        for s, vals in self._ear_by_state.items():
            if vals:
                ear_stats[s] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "min":  round(float(np.min(vals)),  4),
                    "max":  round(float(np.max(vals)),  4),
                    "std":  round(float(np.std(vals)),  4),
                    "n_frames": len(vals)
                }

        summary = {
            "session_duration_s":  round(elapsed, 2),
            "fps_actual":          round(fps_actual, 1),
            "total_frames":        sum(v["n_frames"] for v in ear_stats.values() if v),
            "blink_count":         blink_count,
            "drowsy_events":       drowsy_events,
            "thresholds": {
                "EAR_THRESHOLD":     EAR_THRESHOLD,
                "HEAD_PITCH_THRESH": HEAD_PITCH_THRESH,
                "WARNING_FRAMES":    WARNING_FRAMES,
                "CRITICAL_FRAMES":   CRITICAL_FRAMES,
            },
            "state_durations_s":   {k: round(v, 2) for k, v in self.state_durations.items()},
            "state_pct": {
                k: round(v / max(elapsed, 1) * 100, 1)
                for k, v in self.state_durations.items()
            },
            "ear_stats_by_state":  ear_stats,
        }

        summary_path = os.path.join(self.session_dir, "session_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[Logger] Summary saved → {summary_path}")
        return summary

    def close(self):
        self._frame_file.close()
        self._event_file.close()


# ─── Post-session chart generator ─────────────────────────────────────────────
def generate_charts(session_dir):
    """
    Reads frame_log.csv from session_dir and saves two PNG charts:
      - ear_over_time.png   : EAR + threshold line, state shading, blink markers
      - pitch_over_time.png : head pitch + threshold line, state shading
    These are your Fig. 1 and Fig. 2 for the report.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[Charts] matplotlib not installed — skipping chart generation.")
        return

    frame_csv = os.path.join(session_dir, "frame_log.csv")
    if not os.path.exists(frame_csv):
        return

    times, ears, pitches, states = [], [], [], []
    with open(frame_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["timestamp_s"]))
            ears.append(float(row["ear_smooth"]))
            pitches.append(float(row["pitch_smooth"]))
            states.append(row["state"])

    if not times:
        return

    # Colour map for states
    STATE_COLORS = {
        "ALERT":            "#d4edda",
        "WARNING_EYE":      "#fff3cd",
        "WARNING_HEAD":     "#fde8c8",
        "WARNING_COMBINED": "#ffd5b8",
        "CRITICAL":         "#f8d7da",
    }

    def shade_states(ax, times, states):
        """Draw background shading per state."""
        prev_state = states[0]
        seg_start  = times[0]
        for i in range(1, len(states)):
            if states[i] != prev_state or i == len(states) - 1:
                ax.axvspan(seg_start, times[i],
                           color=STATE_COLORS.get(prev_state, "#ffffff"),
                           alpha=0.4, linewidth=0)
                prev_state = states[i]
                seg_start  = times[i]

    # ── Fig 1: EAR over time ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    shade_states(ax, times, states)
    ax.plot(times, ears, color="#1f77b4", linewidth=0.8, label="Smoothed EAR")
    ax.axhline(EAR_THRESHOLD, color="red", linewidth=1.2, linestyle="--",
               label=f"EAR threshold ({EAR_THRESHOLD})")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Eye Aspect Ratio (EAR)", fontsize=11)
    ax.set_title("Eye Aspect Ratio Over Time with Drowsiness State Annotations", fontsize=12)
    ax.set_ylim(0, max(ears) * 1.15)
    # Legend patches for states
    patches = [mpatches.Patch(color=v, alpha=0.6, label=k) for k, v in STATE_COLORS.items()]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + patches, loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    out_ear = os.path.join(session_dir, "ear_over_time.png")
    plt.savefig(out_ear, dpi=150)
    plt.close()
    print(f"[Charts] Saved {out_ear}")

    # ── Fig 2: Head pitch over time ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    shade_states(ax, times, states)
    ax.plot(times, pitches, color="#ff7f0e", linewidth=0.8, label="Smoothed Head Pitch")
    ax.axhline(HEAD_PITCH_THRESH, color="red", linewidth=1.2, linestyle="--",
               label=f"Pitch threshold ({HEAD_PITCH_THRESH}°)")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Head Pitch (degrees)", fontsize=11)
    ax.set_title("Head Pitch Over Time with Drowsiness State Annotations", fontsize=12)
    patches = [mpatches.Patch(color=v, alpha=0.6, label=k) for k, v in STATE_COLORS.items()]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + patches, loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    out_pitch = os.path.join(session_dir, "pitch_over_time.png")
    plt.savefig(out_pitch, dpi=150)
    plt.close()
    print(f"[Charts] Saved {out_pitch}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    options = FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    pygame.mixer.init()
    audio_cache  = pregenerate_audio()
    alert_engine = OpenAIAlertEngine(audio_cache)
    cap          = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ── NEW: session logger ───────────────────────────────────────────────────
    logger = SessionLogger()

    closed_frames = 0
    head_frames   = 0
    blink_count   = 0
    drowsy_events = 0
    in_blink      = False
    in_drowsy     = False
    state         = "ALERT"
    ear_history   = collections.deque(maxlen=EAR_HISTORY_LEN)
    head_history  = collections.deque(maxlen=HEAD_HISTORY_LEN)
    start_time    = time.time()
    elapsed       = 0
    frame_count   = 0

    print("Drowsiness Detection running — press Q to quit")
    print(f"Using OpenAI TTS (voice: nova) for alerts")

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame        = cv2.flip(frame, 1)
            h, w         = frame.shape[:2]
            elapsed      = time.time() - start_time
            frame_count += 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result    = landmarker.detect(mp_image)

            ear_raw   = 0.30
            pitch_raw = 0.0
            eye_warn  = False
            head_warn = False

            if result.face_landmarks:
                lms = result.face_landmarks[0]

                left_ear   = compute_ear(lms, LEFT_EYE,  w, h)
                right_ear  = compute_ear(lms, RIGHT_EYE, w, h)
                ear_raw    = (left_ear + right_ear) / 2.0
                ear_history.append(ear_raw)
                smooth_ear = np.mean(ear_history)

                pitch_raw = compute_head_pitch(lms, w, h)
                head_history.append(pitch_raw)
                smooth_pitch = np.mean(head_history)

                eye_warn  = smooth_ear   < EAR_THRESHOLD
                head_warn = smooth_pitch > HEAD_PITCH_THRESH

                eye_color = (0, 0, 255) if eye_warn else (0, 255, 0)
                draw_eye_outline(frame, lms, LEFT_EYE,  w, h, eye_color)
                draw_eye_outline(frame, lms, RIGHT_EYE, w, h, eye_color)

                if eye_warn:
                    closed_frames += 1
                    if not in_blink:
                        in_blink = True
                else:
                    if in_blink and closed_frames < WARNING_FRAMES:
                        blink_count += 1
                    in_blink      = False
                    closed_frames = 0

                head_frames = head_frames + 1 if head_warn else 0

                prev_drowsy_events = drowsy_events

                if closed_frames >= CRITICAL_FRAMES or head_frames >= CRITICAL_FRAMES:
                    state = "CRITICAL"
                    if not in_drowsy:
                        drowsy_events += 1
                        in_drowsy = True

                elif closed_frames >= WARNING_FRAMES and head_frames >= WARNING_FRAMES:
                    state = "WARNING_COMBINED"
                    if not in_drowsy:
                        in_drowsy = True
                        drowsy_events += 1

                elif closed_frames >= WARNING_FRAMES:
                    state = "WARNING_EYE"
                    if not in_drowsy:
                        in_drowsy = True
                        drowsy_events += 1

                elif head_frames >= WARNING_FRAMES:
                    state = "WARNING_HEAD"
                    if not in_drowsy:
                        in_drowsy = True
                        drowsy_events += 1
                    if head_frames == WARNING_FRAMES:
                        alert_engine.speak("warning_head")

                else:
                    in_drowsy = False
                    state     = "ALERT"

                if drowsy_events > prev_drowsy_events:
                    alert_engine.speak("warning_head")

                draw_hud(frame, smooth_ear, smooth_pitch, state,
                         blink_count, drowsy_events, elapsed, eye_warn, head_warn)

                # ── NEW: log every frame ──────────────────────────────────
                logger.log_frame(
                    elapsed, ear_raw, smooth_ear,
                    pitch_raw, smooth_pitch,
                    state, closed_frames, head_frames,
                    eye_warn, head_warn, frame
                )

            else:
                cv2.putText(frame, "No face detected — position your face in frame",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
                # Log no-face frames as ALERT with default values
                logger.log_frame(elapsed, 0.30, 0.30, 0.0, 0.0,
                                 "ALERT", 0, 0, False, False, frame)

            cv2.imshow("Driver Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    alert_engine.stop()

    # ── NEW: save summary and generate charts ─────────────────────────────────
    fps_actual = frame_count / max(elapsed, 1)
    summary = logger.save_summary(elapsed, blink_count, drowsy_events, fps_actual)
    logger.close()
    generate_charts(logger.session_dir)

    # ── Print session summary to terminal ─────────────────────────────────────
    print("\n" + "=" * 50)
    print("SESSION SUMMARY")
    print("=" * 50)
    print(f"  Duration       : {int(elapsed)//60:02d}:{int(elapsed)%60:02d}")
    print(f"  FPS (actual)   : {fps_actual:.1f}")
    print(f"  Total frames   : {frame_count}")
    print(f"  Blinks         : {blink_count}")
    print(f"  Drowsy events  : {drowsy_events}")
    print("\n  Time per state:")
    for s, d in summary["state_durations_s"].items():
        pct = summary["state_pct"][s]
        print(f"    {s:<22} {d:>6.1f}s  ({pct:.1f}%)")
    print("\n  EAR stats per state:")
    for s, stats in summary.get("ear_stats_by_state", {}).items():
        print(f"    {s:<22}  mean={stats['mean']:.3f}  min={stats['min']:.3f}  max={stats['max']:.3f}")
    print(f"\n  Session data → {logger.session_dir}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
