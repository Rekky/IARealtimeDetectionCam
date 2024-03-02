import cv2
import mediapipe as mp
from gesture_classifier import RockPaperScissorsClassifier, GestureClassifier
from gesture_classifier.numbers_count_classifier import NumberCount
from hand_detector import HandDetector
from video_capture import VideoCapture


class HandGestureApplication:
    def __init__(self, video_capture: VideoCapture, hand_detector: HandDetector, gesture_classifier: GestureClassifier):
        self.video_capture = video_capture
        self.hand_detector = hand_detector
        self.gesture_classifier = gesture_classifier

    def run(self):
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils

        while True:
            frame = self.video_capture.get_frame()
            if frame is None:
                break

            results = self.hand_detector.process_frame(frame)
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    gesture = self.gesture_classifier.classify(hand_landmarks, handedness)
                    print(f"Gesture detected: {gesture}")

                    # draw in text the gesture of hands
                    cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    # Draw landmarks of the hands
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Show the frame
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('Hand Gesture', frame_bgr)

            if cv2.waitKey(5) & 0xFF == 27:
                break


def init():
    video_capture = VideoCapture()
    hand_detector = HandDetector(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    gesture_classifier = RockPaperScissorsClassifier()
    # gesture_classifier = NumberCount()
    app = HandGestureApplication(video_capture, hand_detector, gesture_classifier)
    app.run()


if __name__ == '__main__':
    init()