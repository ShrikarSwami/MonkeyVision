# 🐒 MonkeyVision

Real-time hand + face gesture fun in Python/OpenCV + MediaPipe.  
Point 👉 to cue the “say” monkey, touch index to mouth 🤫 to show the “think” monkey — with sound!

Repo: https://github.com/ShrikarSwami/MonkeyVision  
Demo video: https://youtu.be/Cj1oETl1BUY

---

## 🎬 Demo

- POINT 👉 = “say” monkey
- THINK 🤫 (index fingertip near mouth) = “think” monkey

If you’re reading this on GitHub, you’ll also see GIFs in `docs/` when added.  
Demo video again (because hype): https://youtu.be/Cj1oETl1BUY

---

## ✨ Features

- 🖐️ One-hand gesture detection  
  - **POINT**: index up (others down)  
  - **THINK**: index fingertip near mouth (6% of frame diagonal)
- 🙂 Face mesh → mouth center + smart head box
- 🧩 Alpha-blended PNG monkeys overlayed on your webcam feed
- 🔊 One-shot music trigger the **first** time you hit POINT
- 🖥️ Cross-platform camera backend selection (macOS, Windows, Linux)
- ⚡ HUD with FPS + current gesture label

> ✅ All image/audio assets (`monkey_*.png`, `music.mp3`) are included in the repo.  
> The script looks for them right next to `webcam.py` — nothing extra to download.

---

## 🧰 Requirements

- **Python 3.11** (MediaPipe wheels aren’t out yet for 3.13 at time of writing)
- macOS / Windows / Linux with a webcam
- Internet only for `pip install` (no cloud at runtime)

---

## 🚀 TL;DR — Quick Start

1) **Download / Clone**

- Easiest: click the green **Code** button → **Download ZIP** (default branch is cross-platform).  
  Unzip, then `cd` into the project folder.

- Or clone:
  ```bash
  git clone https://github.com/ShrikarSwami/MonkeyVision.git
  cd MonkeyVision
