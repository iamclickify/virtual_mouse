# Virtual Mouse using Hand Gestures

Control your computer without touching a mouse!

This project uses computer vision to track hand movements in real-time and converts them into mouse actions like cursor movement and clicks.

---

## 📌 Features
- 🖐️ Real-time hand tracking
- 🖱️ Move cursor using finger movement
- 👆 Perform click actions using gestures
- ⚡ Smooth and responsive interaction

---

## 🧠 How it Works
- Uses a webcam to capture live video feed
- Detects hand landmarks using a hand tracking model
- Maps finger positions to screen coordinates
- Recognizes specific gestures for actions like clicking

---

## ⚙️ Tech Stack
- Python
- OpenCV
- MediaPipe
- PyAutoGUI

---

## 🎥 Demo
(Add a short demo video or GIF here)

---

## 🚧 Challenges Faced
- Mapping camera coordinates to screen space accurately
- Reducing jitter for smoother cursor movement
- Handling different lighting conditions for reliable detection

---

## 🚀 Future Improvements
- Add gesture-based volume/brightness control
- Improve tracking accuracy
- Add multi-hand support
- Build a small UI overlay for better feedback

---

## 📂 How to Run
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install opencv-python mediapipe pyautogui
   ```
3. Run the script:
   ```bash
   python main.py
   ```

---
Make sure to change ppt path!
ppt_path = r"C:\Users\admin\OneDrive\Documents\virtual_mouse.pptx"

## 💡 Learnings
- Basics of computer vision and hand tracking
- Working with real-time systems
- Translating physical gestures into digital actions

---

## 🙌 Acknowledgment
This project was built as part of learning computer vision and gesture-based interaction systems.
