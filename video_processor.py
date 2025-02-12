import cv2

class VideoProcessor:
    def __init__(self, source):
        """
        Inicializa la fuente de video.
        :param source: Ruta del archivo de video o un entero para la webcam (e.g., 0 para cámara principal).
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    def get_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()
