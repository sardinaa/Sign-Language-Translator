import numpy as np
import cv2
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor
from .video_processor import VideoProcessor

class LandmarkExtractor:
    def __init__(self, filtered_hand, filtered_pose, filtered_face):
        self.filtered_hand = filtered_hand
        self.filtered_pose = filtered_pose
        self.filtered_face = filtered_face

        # Inicialización de modelos de MediaPipe
        self.hands = mp.solutions.hands.Hands()
        self.pose = mp.solutions.pose.Pose()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    def get_frame_landmarks(self, frame):
        """
        Procesa un único frame y extrae landmarks de manos, pose y rostro.
        """
        num_landmarks = len(self.filtered_hand) * 2 + len(self.filtered_pose) + len(self.filtered_face)
        all_landmarks = np.zeros((num_landmarks, 3))

        # Métodos internos para procesar manos, pose y rostro
        def get_hands(frame):
            results_hands = self.hands.process(frame)
            if results_hands.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                    if results_hands.multi_handedness[i].classification[0].index == 0:  # Mano derecha
                        all_landmarks[:len(self.filtered_hand), :] = np.array(
                            [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                        )[self.filtered_hand]
                    else:  # Mano izquierda
                        all_landmarks[len(self.filtered_hand):len(self.filtered_hand) * 2, :] = np.array(
                            [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                        )[self.filtered_hand]

        def get_pose(frame):
            results_pose = self.pose.process(frame)
            if results_pose.pose_landmarks:
                all_landmarks[len(self.filtered_hand) * 2:len(self.filtered_hand) * 2 + len(self.filtered_pose), :] = np.array(
                    [(lm.x, lm.y, lm.z) for lm in results_pose.pose_landmarks.landmark]
                )[self.filtered_pose]

        def get_face(frame):
            results_face = self.face_mesh.process(frame)
            if results_face.multi_face_landmarks:
                all_landmarks[len(self.filtered_hand) * 2 + len(self.filtered_pose):, :] = np.array(
                    [(lm.x, lm.y, lm.z) for lm in results_face.multi_face_landmarks[0].landmark]
                )[self.filtered_face]

        # Procesar en paralelo usando hilos
        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.submit(get_hands, frame)
            executor.submit(get_pose, frame)
            executor.submit(get_face, frame)

        return all_landmarks  

    def get_video_landmarks(self, video_source, start_frame=1, end_frame=None):
            """
            Procesa un video completo o la webcam y devuelve los landmarks de cada frame.
            :param video_source: Ruta del video o un entero para webcam (e.g., 0 para cámara principal).
            :param start_frame: Frame inicial a procesar.
            :param end_frame: Frame final a procesar (opcional).
            :return: Array con los landmarks de todos los frames procesados.
            """
            video_processor = VideoProcessor(video_source)

            # Ajustar frames inicial y final si es un archivo de video
            if video_processor.total_frames:
                start_frame = max(1, start_frame)
                if end_frame is None or end_frame < 0 or end_frame > video_processor.total_frames:
                    end_frame = video_processor.total_frames
            else:
                # Webcam no tiene total de frames, procesamos en tiempo real
                start_frame, end_frame = 1, None

            num_landmarks = len(self.filtered_hand) * 2 + len(self.filtered_pose) + len(self.filtered_face)
            all_frame_landmarks = np.zeros((end_frame - start_frame + 1, num_landmarks, 3))

            frame_index = 1
            while frame_index <= end_frame if end_frame else True:
                ret, frame = video_processor.get_frame()
                if not ret:
                    break

                if frame_index >= start_frame:
                    frame.flags.writeable = False
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_landmarks = self.get_frame_landmarks(frame)
                    all_frame_landmarks[frame_index - start_frame] = frame_landmarks

                frame_index += 1

            video_processor.release()
            self.hands.reset()
            self.pose.reset()
            self.face_mesh.reset()
            return np.array(all_frame_landmarks)