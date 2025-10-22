# 🐒 MonkeyVision

Real-time hand + face gesture fun in Python/OpenCV + MediaPipe.  
Point 👉 to cue the “say” monkey, touch index to mouth 🤫 to show the “think” monkey — with sound!

Repo: https://github.com/ShrikarSwami/MonkeyVision

---

## 🎬 Demo

[![MonkeyVision Demo (YouTube)](https://img.youtube.com/vi/Cj1oETl1BUY/0.jpg)](https://youtu.be/Cj1oETl1BUY "MonkeyVision Demo")

![POINT demo](docs/point.gif)
![THINK demo](docs/think.gif)

> Tip: short MP4s → convert to GIFs for GitHub, and stash them in `docs/`.

---

## ✨ Features

- 🖐️ One-hand gesture detection (index up = **POINT**; index near mouth = **THINK**)
- 🙂 Face mesh for mouth + head box; overlay sits above your head
- 🧩 Alpha-blended PNG monkeys composited on your camera
- 🔊 One-shot music trigger the first time you hit POINT
- 🖥️ Cross-platform camera backend selection (macOS, Windows, Linux)
- ⚡ HUD with FPS + current gesture

---

## 📦 Requirements

- **Python 3.10–3.11** (MediaPipe wheels are reliable here)
- macOS / Windows / Linux with a webcam
- Internet to `pip install` stuff

---

## 🚀 Quick Start

### 1) Clone
# (Terminal)
git clone https://github.com/ShrikarSwami/MonkeyVision.git
cd MonkeyVision

### 2) Create a virtual environment & install deps

# macOS / Linux
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt

# Windows (PowerShell)
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install -r requirements.txt

### 3) Drop assets in the repo folder
# Images (transparent PNGs work best):
#   monkey_neutral.png, monkey_say.png, monkey_think.png
# Sound:
#   music.mp3
#
# Put them in the REPO ROOT (same folder as webcam.py),
# or pass a custom music path with --music.

### 4) Run it

# Default camera, auto backend (recommended)
python webcam.py

# Different camera index or resolution
python webcam.py --camera 1 --width 1280 --height 720

# Force a specific backend if the default acts weird:
#   macOS   → --backend avfoundation
#   Windows → --backend dshow   (or --backend msmf)
#   Linux   → --backend v4l2    (or --backend gstreamer)
python webcam.py --backend avfoundation

# Custom celebration music
python webcam.py --music ./my_song.mp3

---

## 🧠 Gestures (how to “monkey”)

# POINT 👉 : Raise your index finger (others down).
#           → Shows “say” monkey + plays music once per session.
# THINK 🤫 : Bring index fingertip near your mouth.
#           → Shows “think” monkey.

# The overlay is auto-sized to your face box and plopped just above your head.

---

## 🧯 Troubleshooting (quick)

# Camera won’t open
#   • Try a different backend:
#       macOS → --backend avfoundation
#       Windows → --backend dshow (or msmf)
#       Linux → --backend v4l2
#   • Try another camera index: --camera 1 (or 2, 3…)

# No audio
#   • Ensure pygame installed and speakers not muted.
#   • Use a valid path to music.mp3 (absolute path if in doubt):
#       --music /full/path/to/music.mp3

# Low FPS
#   • Lower resolution: --width 640 --height 480

# Virtual env hiccups
#   • Deactivate + reactivate the venv.
#   • On Windows you might need to allow scripts:
#       Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

---

## 🌲 Branches

# main          → stable macOS-first version
# crossplatform → recommended for everyone (auto-backend + CLI flags)

# Switch branches:
git switch crossplatform
# or:
git switch main

---

## 🛠️ Dev Notes

# • Uses OpenCV’s named window & alpha-blend overlay for PNGs.
# • MediaPipe Hands + Face Mesh drive gesture logic:
#     - finger up = index TIP.y < PIP.y
#     - THINK = (index tip) near (mouth center)
# • Smoothing via a tiny deque history for stable labels.
# • Music plays on the first transition into POINT (one-shot).

---

## 🤝 Contributing

# PRs welcome! Please:
# • Keep commits small and messages clear.
# • Test on at least one OS (Windows/macOS/Linux).
# • Attach a quick GIF if you change visuals.

---

## 📄 License

# MIT — go build something silly and awesome.

---

## 🙌 Thanks

# OpenCV, MediaPipe, pygame, and your patient webcam.
