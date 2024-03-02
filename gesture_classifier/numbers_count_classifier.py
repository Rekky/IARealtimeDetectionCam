import mediapipe as mp
import numpy as np

from gesture_classifier import GestureClassifier


class NumberCount(GestureClassifier):
    def calculate_distance(self, landmark1, landmark2):
        return np.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)

    def classify(self, hand_landmarks, handedness):
        thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
        wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]

        thumb_distance = self.calculate_distance(thumb_tip, wrist)
        index_distance = self.calculate_distance(index_tip, wrist)
        middle_distance = self.calculate_distance(middle_tip, wrist)
        ring_distance = self.calculate_distance(ring_tip, wrist)
        pinky_distance = self.calculate_distance(pinky_tip, wrist)


        if index_distance > 0.5 and thumb_distance < 0.5 and middle_distance < 0.5 and ring_distance < 0.5 and pinky_distance < 0.5:
            return "ONE"
        if index_distance > 0.5 and thumb_distance < 0.5 and middle_distance > 0.5 and ring_distance < 0.5 and pinky_distance < 0.5:
            return "TWO"
        if index_distance > 0.5 and thumb_distance < 0.5 and middle_distance > 0.5 and ring_distance > 0.5 and pinky_distance < 0.5:
            return "THREE"
        if index_distance > 0.5 and thumb_distance < 0.5 and middle_distance > 0.5 and ring_distance > 0.5 and pinky_distance > 0.5:
            return "FOUR"
        if index_distance > 0.5 and thumb_distance > 0.5 and middle_distance > 0.5 and ring_distance > 0.5 and pinky_distance > 0.5:
            return "FIVE"
        else:
            return "BAD"
