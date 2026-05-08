# Driver_drowsines_detection_computer_vision
# 🚗 Driver Drowsiness Detection with AI Voice Alerts

A real-time driver drowsiness detection system using computer vision and AI-powered text-to-speech alerts. It monitors eye closure (EAR) and head drop via your webcam, escalates alerts through spoken warnings using OpenAI TTS, and logs every session with charts for post-drive analysis.

---

## 📸 Demo

> The system overlays a live HUD on your webcam feed showing EAR score, head pitch, current alert state, blink count, and session timer. Alerts are spoken aloud using a natural AI voice.

---

## ✨ Features

- 👁️ **Eye Aspect Ratio (EAR)** tracking to detect eye closure
- 🙆 **Head pitch estimation** to detect nodding/head drop
- 🔊 **AI voice alerts** via OpenAI TTS (voice: Nova), pre-generated at startup
- 📊 **Post-session charts** — EAR and head pitch over time with state shading
- 🗂️ **Session logging** — per-frame CSV, state transition events, and a JSON summary
- 📸 **Auto screenshots** saved on every state change
- 🎯 **4 alert levels**: Warning (Eye), Warning (Head), Warning (Combined), Critical

---

## 🛠️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

**Never hardcode your API key.** Set it as an environment variable:

```bash
# macOS / Linux
export OPENAI_API_KEY="your-key-here"

# Windows (Command Prompt)
set OPENAI_API_KEY=your-key-here

# Windows (PowerShell)
$env:OPENAI_API_KEY="your-key-here"
```

### 4. Run

```bash
python drowsiness_detect_tts.py
```

The MediaPipe face landmark model (~30MB) will be downloaded automatically on first run. All alert audio clips are pre-generated via OpenAI TTS before detection begins.

Press **Q** to quit. A session summary will be printed and all data saved automatically.

---

## 📦 Requirements / Packages

| Package | Purpose |
|---|---|
| `opencv-python` | Webcam capture, frame display, HUD drawing |
| `mediapipe` | Face landmark detection (468 landmarks) |
| `numpy` | EAR smoothing, pitch computation |
| `scipy` | Euclidean distance for EAR calculation |
| `openai` | TTS audio generation (model: `tts-1`, voice: `nova`) |
| `pygame` | MP3 audio playback for voice alerts |
| `matplotlib` | Post-session EAR and head pitch charts |

Install all at once:

```bash
pip install opencv-python mediapipe numpy scipy openai pygame matplotlib
```

Or use the `requirements.txt`:

```
opencv-python
mediapipe
numpy
scipy
openai
pygame
matplotlib
```

---

## 📁 Output Structure

Each run creates a timestamped session folder:

```
session_YYYYMMDD_HHMMSS/
├── frame_log.csv          # Per-frame: EAR, pitch, state, warnings
├── event_log.csv          # State transitions with latency info
├── session_summary.json   # Aggregated stats (blinks, drowsy events, state %)
├── ear_over_time.png      # Chart: EAR with state shading
├── pitch_over_time.png    # Chart: Head pitch with state shading
└── screenshots/
    └── 001_ALERT_to_WARNING_EYE_12.3s.png   # Frame saved on each state change
```

---

## ⚙️ Detection Parameters

| Parameter | Default | Description |
|---|---|---|
| `EAR_THRESHOLD` | `0.22` | EAR below this = eyes closing |
| `HEAD_PITCH_THRESH` | `15.0°` | Pitch above this = head drop |
| `WARNING_FRAMES` | `20` | Frames before a warning alert |
| `CRITICAL_FRAMES` | `40` | Frames before a critical alert |
| `EAR_HISTORY_LEN` | `10` | Smoothing window for EAR |
| `HEAD_HISTORY_LEN` | `10` | Smoothing window for head pitch |

---

## 🔒 Security Note

Never commit your OpenAI API key to version control. Use environment variables or a `.env` file with `python-dotenv`. Add `.env` to your `.gitignore`.

---

## 📄 License

MIT License — feel free to use and adapt.
