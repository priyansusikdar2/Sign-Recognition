# 🧠 Real-Time Sign Language Detection with Voice Output

This project uses **OpenCV**, **MediaPipe**, and **pyttsx3** to detect and recognize hand gestures representing sign language alphabets in real time via webcam, and converts them into **spoken output** using a Text-to-Speech engine.

---

---

## ✅ Features

- ✋ Real-time hand and finger tracking using MediaPipe
- 🔡 Maps gestures to specific sign language alphabets
- 🗣️ Speaks the detected letters out loud using pyttsx3
- 📋 Buffers detected letters into phrases
- ⏱️ Logs recognized letters with timestamps
- 💬 Press `C` to clear the phrase buffer
- 🔍 Gesture stability checking to avoid flickering output

---

## 🧪 Requirements

Install the dependencies using pip:

```bash
pip install opencv-python mediapipe pyttsx3

🚀 How to Run
Connect a webcam.

Run the main script:

bash
Copy
Edit
python sign_language_detector.py

Make hand gestures in front of the camera.

The detected letter will:

Be printed on screen

Added to the current phrase

Spoken aloud using Text-to-Speech

Press:

ESC to exit

C to clear the phrase buffer

🔤 Gesture-to-Letter Mapping
Gesture Description	Finger States	Mapped Letter
All fingers folded	[0, 0, 0, 0, 0]	A
All fingers extended	[1, 1, 1, 1, 1]	5
Index only up	[0, 1, 0, 0, 0]	D
Index + middle up	[0, 1, 1, 0, 0]	U
Four fingers up (no thumb)	[0, 1, 1, 1, 1]	B
Thumb and pinky up	[1, 0, 0, 0, 1]	Y (custom)

You can easily extend gesture_map in the script to add more gestures.

