# 🐒 MonkeyVision

Real-time hand + face gesture fun in Python/OpenCV + MediaPipe.  
Point 👉 to cue the “say” monkey, touch index to mouth 🤫 to show the “think” monkey — with sound!

https://github.com/ShrikarSwami/MonkeyVision

---

## Demo

![POINT demo](docs/point.gif)
![THINK demo](docs/think.gif)

> Tip: Drop short MP4s in `docs/` and embed as GIFs (GitHub renders GIFs best).  
> You can also add a YouTube link or GitHub asset for your dorm demo video.

---

## Features
- 🖐️ One-hand gesture detection (index up = POINT, index to mouth = THINK)
- 🙂 Face mesh for mouth + head box
- 🧩 Alpha-blended PNG monkeys overlayed on your video
- 🔊 One-shot music trigger the first time you hit POINT
- 🖥️ Cross-platform camera backend (macOS, Windows, Linux)
- ⚡ HUD with FPS + current gesture

---

## Requirements

- **Python 3.11** (MediaPipe doesn’t provide wheels for 3.13 yet)
- macOS/Windows/Linux with a webcam

Install from the repo root:

```bash
# macOS
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt

# Windows (PowerShell)
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install -r requirements.txt
