import cv2


class VideoCapture:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)

    def get_frame(self):
        success, frame = self.cap.read()
        if success:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            return None
