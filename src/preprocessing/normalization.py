import numpy as np

class LandmarkNormalizer:
    def __init__(self, indices, threshold=0.5):
        """
        Inicializa el normalizador con configuraciones específicas.
        
        Args:
            indices (dict): Diccionario con los índices globales necesarios (hombros, nariz, muñecas, etc.).
            threshold (float): Umbral para considerar una secuencia válida (proporción de datos válidos).
        """
        self.indices = indices
        self.threshold = threshold

    def detect_invalid_sequences(self, landmarks):
        """
        Detecta si una secuencia tiene suficientes datos válidos en los landmarks críticos.
        
        Args:
            landmarks (np.ndarray): Secuencia de landmarks de forma (frames, landmarks, coordenadas).

        Returns:
            bool: True si la secuencia es válida, False en caso contrario.
        """
        total_frames = landmarks.shape[0]
        valid_frames = 0
        for idx in self.indices['critical']:
            coords = landmarks[:, idx, :]
            valid_frames += np.sum(~np.all(coords == 0, axis=1))
        return (valid_frames / total_frames) >= self.threshold

    @staticmethod
    def normalize_hand(hand_landmarks, wrist_coords):
        """
        Normaliza las manos en relación a su bounding box local.
        
        Args:
            hand_landmarks (np.ndarray): Coordenadas de la mano.
            wrist_coords (np.ndarray): Coordenadas de la muñeca.

        Returns:
            np.ndarray: Landmarks de la mano normalizados.
        """
        if np.all(hand_landmarks == 0):
            return hand_landmarks

        x_vals = hand_landmarks[:, 0]
        y_vals = hand_landmarks[:, 1]

        min_x, max_x = np.min(x_vals), np.max(x_vals)
        min_y, max_y = np.min(y_vals), np.max(y_vals)

        box_width = max_x - min_x if max_x > min_x else 1e-6
        box_height = max_y - min_y if max_y > min_y else 1e-6

        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0

        hand_norm_x = (x_vals - center_x) / box_width
        hand_norm_y = (y_vals - center_y) / box_height

        hand_scaled_x = wrist_coords[0] + hand_norm_x / 6.0
        hand_scaled_y = wrist_coords[1] + hand_norm_y / 7.0

        return np.stack([hand_scaled_x, hand_scaled_y], axis=-1)

    def normalize_frame(self, landmarks, avg_head_unit):
        """
        Normaliza un frame individual en relación al espacio de signing.
        
        Args:
            landmarks (np.ndarray): Coordenadas de los landmarks en el frame.
            avg_head_unit (float): Tamaño promedio de la cabeza.

        Returns:
            np.ndarray: Landmarks normalizados en el frame.
        """
        indices = self.indices
        ls, rs = landmarks[indices['shoulder_left'], :2], landmarks[indices['shoulder_right'], :2]

        if np.all(ls == 0) or np.all(rs == 0):
            return landmarks[:, :2]

        nose = landmarks[indices['nose'], :2]
        le = landmarks[indices['left_eye'], :2] if not np.all(landmarks[indices['left_eye'], :2] == 0) else nose

        head_unit = avg_head_unit
        body_top = le[1] - 0.5 * head_unit
        body_bottom = body_top + 7.0 * head_unit
        body_left = nose[0] - 3 * head_unit
        body_right = nose[0] + 3 * head_unit

        body_width = body_right - body_left
        body_height = body_bottom - body_top

        normalized_body_x = (landmarks[:, 0] - body_left) / body_width
        normalized_body_y = (landmarks[:, 1] - body_top) / body_height

        final_landmarks = np.zeros((landmarks.shape[0], 2))
        final_landmarks[:, 0] = normalized_body_x
        final_landmarks[:, 1] = normalized_body_y

        left_wrist_norm = np.array([(landmarks[indices['left_wrist_pose'], 0] - body_left) / body_width,
                                    (landmarks[indices['left_wrist_pose'], 1] - body_top) / body_height])
        right_wrist_norm = np.array([(landmarks[indices['right_wrist_pose'], 0] - body_left) / body_width,
                                     (landmarks[indices['right_wrist_pose'], 1] - body_top) / body_height])

        left_hand_norm = final_landmarks[indices['left_hand']]
        right_hand_norm = final_landmarks[indices['right_hand']]

        final_landmarks[indices['left_hand']] = self.normalize_hand(left_hand_norm, left_wrist_norm)
        final_landmarks[indices['right_hand']] = self.normalize_hand(right_hand_norm, right_wrist_norm)

        final_landmarks[indices['left_wrist_pose']] = left_wrist_norm
        final_landmarks[indices['right_wrist_pose']] = right_wrist_norm

        return final_landmarks

    def normalize_sequence(self, landmarks_sequence):
        """
        Normaliza una secuencia completa de landmarks.
        
        Args:
            landmarks_sequence (np.ndarray): Secuencia de landmarks de forma (frames, landmarks, coordenadas).

        Returns:
            np.ndarray: Secuencia normalizada.
        """
        frames, _, _ = landmarks_sequence.shape

        if not self.detect_invalid_sequences(landmarks_sequence):
            print("Secuencia descartada: demasiados valores faltantes en landmarks críticos.")
            return None

        head_units = [
            np.linalg.norm(landmarks_sequence[f, self.indices['shoulder_right'], :2] -
                           landmarks_sequence[f, self.indices['shoulder_left'], :2]) / 2.0
            for f in range(frames)
            if not np.all(landmarks_sequence[f, self.indices['shoulder_left'], :2] == 0)
        ]
        avg_head_unit = np.mean(head_units) if head_units else 1.0

        normalized_seq = np.array([self.normalize_frame(landmarks_sequence[f], avg_head_unit) for f in range(frames)])
        return normalized_seq
