# 🐒 MonkeyVision

Real-time hand + face gesture fun in Python/OpenCV + MediaPipe.  
Point 👉 to cue the “say” monkey, touch index to mouth 🤫 to show the “think” monkey — with sound!

Demo video: https://youtu.be/Cj1oETl1BUY

---

## Features

- 🖐️ One-hand gesture detection  
  - **POINT** = index up (others down)  
  - **THINK** = index fingertip near mouth
- 🙂 Face mesh for mouth + auto “head box”
- 🧩 Alpha-blended PNG monkeys overlaid on your webcam (**PNG files are already in this repo**)
- 🔊 One-shot music the first time you hit POINT (**music.mp3 is already in this repo**)
- 🖥️ Cross-platform camera backend selection (macOS / Windows / Linux)
- ⚡ Minimal HUD with FPS + current gesture

---

## Requirements

- **Python 3.10–3.11** (MediaPipe wheels are reliable here)
- A webcam (built-in or USB)

Everything else is pinned in `requirements.txt`.

---

## Quick Start (copy–paste these)

### 1) Clone
git clone https://github.com/ShrikarSwami/MonkeyVision.git
cd MonkeyVision

### 2) Create & activate a virtual env

# macOS / Linux (bash or zsh)
python3.11 -m venv .venv311
source .venv311/bin/activate

# Windows (PowerShell)
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1

### 3) Install deps
pip install -r requirements.txt

### 4) Run it
# Auto-picks a good backend per OS (AVFoundation on macOS, DirectShow on Windows, V4L2 on Linux)
python webcam.py

The app opens your webcam, draws the HUD, and overlays the included assets:

- monkey_neutral.png  
- monkey_say.png  
- monkey_think.png  
- music.mp3

No extra downloads needed — they’re already in the repo.

---

## Command-line Options (nice to have)

# Choose a camera index (0 is default)
python webcam.py --camera 1

# Force a specific backend if auto is weird:
#   macOS:   avfoundation
#   Windows: dshow or msmf
#   Linux:   v4l2 or gstreamer
python webcam.py --backend dshow

# Change resolution (defaults: 1920x1080)
python webcam.py --width 1280 --height 720

# Use a different celebration track (defaults to ./music.mp3 which is included)
python webcam.py --music path/to/your_song.mp3

Backends accepted: auto | avfoundation | dshow | msmf | v4l2 | gstreamer | any  
- `auto` = smart pick by OS  
- `any`  = let OpenCV decide

---

## Gestures (how to trigger)

- **POINT** 👉  
  Raise your **index** finger; keep middle/ring/pinky down.  
  Shows the “say” monkey + plays music **once** per run.

- **THINK** 🤫  
  Move your **index fingertip near your mouth** (≈ 6% of frame diagonal).  
  Shows the “think” monkey.

- Otherwise: neutral monkey.

---

## Troubleshooting (90-second fixes)

- **Black/empty camera or wrong cam**
  - Try another index: `--camera 1`, `--camera 2`
  - Force backend:
    - macOS → `--backend avfoundation`
    - Windows → `--backend dshow` (or `--backend msmf`)
    - Linux → `--backend v4l2`

- **No audio**
  - Check system output device; pygame uses the default.
  - Try a different MP3: `--music somefile.mp3`

- **Low FPS**
  - Lower resolution: `--width 1280 --height 720`
  - Close other camera apps.

---

## Repo Layout

MonkeyVision/
├─ webcam.py               # The app (cross-platform backends + CLI)
├─ requirements.txt        # Pinned deps
├─ monkey_neutral.png      # Included overlay art
├─ monkey_say.png          # Included overlay art
├─ monkey_think.png        # Included overlay art
├─ music.mp3               # Included sample track (POINT celebration)
├─ README.md               # This file
└─ LICENSE

---

## License

MIT — go wild (and maybe keep the monkeys).

---

## Thanks

OpenCV, MediaPipe, pygame, and your patient webcam.
