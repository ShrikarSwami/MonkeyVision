# 🐒 MonkeyVision

Real-time hand + face gesture fun with Python, OpenCV, and MediaPipe.  
Raise your **index finger** to cue the **“say” monkey**; bring your **index to your mouth** to show the **“think” monkey**—with an optional celebration sound.

> Repo: https://github.com/ShrikarSwami/MonkeyVision

---

## Demo

[![MonkeyVision Demo (YouTube)](https://img.youtube.com/vi/Cj1oETl1BUY/0.jpg)](https://youtu.be/Cj1oETl1BUY "MonkeyVision Demo")

![POINT demo](docs/point.gif)
![THINK demo](docs/think.gif)

---

## Features

- 🖐️ One-hand gesture detection (index up = **POINT**; index near mouth = **THINK**)
- 🙂 Face mesh for mouth + head box; overlay sits above your head
- 🧩 Alpha-blended PNG monkeys composited on your camera
- 🔊 One-shot music trigger the first time you hit POINT
- 🖥️ Cross-platform camera backend selection (macOS, Windows, Linux)
- ⚡ HUD with FPS + current gesture

---

## Requirements

- **Python 3.10–3.11** (MediaPipe wheels are reliable here)
- macOS / Windows / Linux with a webcam

---

## Quick Start

### 1) Clone
```bash
git clone https://github.com/ShrikarSwami/MonkeyVision.git
cd MonkeyVision
