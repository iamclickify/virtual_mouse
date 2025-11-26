import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import speech_recognition as sr
import pyttsx3
import os
import time

# ======================
#  Voice Setup
# ======================
r = sr.Recognizer()
engine = pyttsx3.init()
engine.setProperty('rate', 170)

def speak(text):
    print("💬 " + text)
    engine.say(text)
    engine.runAndWait()

# ======================
#  Hand Gesture Config
# ======================
W_CAM, H_CAM = 640, 480
FRAME_REDUCTION = 100
SMOOTHING = 4
SWIPE_THRESHOLD = 50  # Minimum movement in X direction for swipe detection

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

prev_x = 0
gesture_cooldown = 0  # To avoid multiple slide turns per swipe

# ======================
#  Voice Command
# ======================
def listen_for_presentation():
    """Waits for user to say 'open presentation' or 'open PowerPoint'."""
    while True:
        try:
            with sr.Microphone() as source:
                speak("Please say 'open presentation' to start.")
                print("\n🎙️ Listening...")
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
                print("🧠 Recognizing...")
                command = r.recognize_google(audio).lower()
                print(f"👉 You said: {command}")
                if "presentation" in command or "powerpoint" in command:
                    return True
                else:
                    speak("Please say 'open presentation' to continue.")
        except sr.UnknownValueError:
            speak("I didn’t catch that, please repeat.")
        except sr.WaitTimeoutError:
            print("⌛ Timeout, listening again...")
        except sr.RequestError:
            speak("Speech service unavailable.")
            break

# ======================
#  Open PowerPoint
# ======================
def open_ppt():
    ppt_path = r"C:\Users\admin\OneDrive\Documents\presentation.pptx"  
    if os.path.exists(ppt_path):
        os.startfile(ppt_path)
        speak("Opening your presentation.")
        time.sleep(5)
        speak("You can now use your hand to turn slides.")
        return True
    else:
        speak("Presentation file not found.")
        return False

# ======================
#  Gesture Controller
# ======================
def gesture_control():
    """Detects left and right hand movements to change slides."""
    global prev_x, gesture_cooldown
    cap = cv2.VideoCapture(0)
    cap.set(3, W_CAM)
    cap.set(4, H_CAM)
    speak("Gesture control activated. Swipe right or left to change slides. Press 'q' to exit.")

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

                # Get the x position of index finger
                index_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x_pos = int(index_tip.x * W_CAM)

                # Detect swipe motion (with cooldown to prevent repeat triggers)
                if gesture_cooldown == 0:
                    if x_pos - prev_x > SWIPE_THRESHOLD:
                        pyautogui.press("right")
                        speak("Next slide.")
                        gesture_cooldown = 10
                    elif prev_x - x_pos > SWIPE_THRESHOLD:
                        pyautogui.press("left")
                        speak("Previous slide.")
                        gesture_cooldown = 10

                prev_x = x_pos

        # Decrease cooldown
        if gesture_cooldown > 0:
            gesture_cooldown -= 1

        cv2.imshow("🖐️ Gesture Slide Controller", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    speak("Gesture control stopped.")


if __name__ == "__main__":
    speak("Hello! I am your presentation assistant.")
    if listen_for_presentation():
        if open_ppt():
            gesture_control()
        else:
            speak("Unable to start presentation mode.")
