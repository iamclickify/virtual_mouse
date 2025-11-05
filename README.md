# Virtual Mouse Controller using Hand Gestures 🖐️

This project allows you to **control your computer’s mouse using hand gestures** through a webcam.  
It uses **MediaPipe** for hand tracking and **PyAutoGUI** for controlling the mouse cursor and performing click actions.

---

## 🚀 Features

- Move the mouse pointer by moving your **index finger**.
- Perform **left click** using **thumb + index pinch gesture**.
- Perform **right click** using **thumb + middle finger pinch gesture**.
- Smooth pointer movement and reduced accidental clicks using a **state machine** and **distance thresholds**.
- Adjustable sensitivity and frame reduction area.

---

## 🧠 Tech Stack

- **Python 3.8+**
- **OpenCV** – for real-time webcam input and visualization  
- **MediaPipe** – for hand and finger landmark detection  
- **PyAutoGUI** – for mouse cursor movement and click events  
- **NumPy** – for numerical operations  

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/Citradev/virtual-mouse-controller.git
cd virtual-mouse-controller
