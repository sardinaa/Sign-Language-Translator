import numpy as np

class DataAugmentor:
    def __init__(self, hand_num, pose_num, default_config=None):
        """
        Inicializa la clase de aumentación de datos.
        
        Args:
            hand_num (int): Número de landmarks por mano.
            pose_num (int): Número de landmarks de la pose.
            default_config (dict): Configuraciones por defecto para las aumentaciones.
        """
        self.HAND_NUM = hand_num
        self.POSE_NUM = pose_num

        # Configuración predeterminada
        self.default_config = default_config or {
            "rotation_axes": ["x", "y", "z"],
            "rotation_angles": {"x": (-10, 10), "y": (-5, 5), "z": (-20, 20)},
            "zoom_factor": (0.9, 1.1),
            "x_shift": (-0.1, 0.1),
            "y_shift": (-0.1, 0.1),
        }

    @staticmethod
    def rotate(data, rotation_matrix):
        frames, landmarks, _ = data.shape
        center = np.array([0.5, 0.5, 0])

        non_zero = np.argwhere(np.any(data[:, :, :2] != 0, axis=2))

        data = data.reshape(-1, 3)
        data[non_zero[:, 0] * landmarks + non_zero[:, 1]] -= center
        data[non_zero[:, 0] * landmarks + non_zero[:, 1]] = (
            np.dot(data[non_zero[:, 0] * landmarks + non_zero[:, 1]], rotation_matrix.T)
        )
        data[non_zero[:, 0] * landmarks + non_zero[:, 1]] += center
        return data.reshape(frames, landmarks, 3)

    @staticmethod
    def generate_rotation_matrix_per_axis(rotation_axes, rotation_angles):
        rotation_matrix = np.eye(3)
        for axis in rotation_axes:
            angle_range = rotation_angles.get(axis, (0, 0))
            angle_deg = np.random.uniform(*angle_range)
            angle = np.radians(angle_deg)

            if axis == "x":
                R_x = np.array([
                    [1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)]
                ])
                rotation_matrix = rotation_matrix @ R_x
            elif axis == "y":
                R_y = np.array([
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]
                ])
                rotation_matrix = rotation_matrix @ R_y
            elif axis == "z":
                R_z = np.array([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]
                ])
                rotation_matrix = rotation_matrix @ R_z
        return rotation_matrix

    @staticmethod
    def zoom(data, factor):
        center = np.array([0.5, 0.5])

        non_zero = np.argwhere(np.any(data[:, :, :2] != 0, axis=2))
        data[non_zero[:, 0], non_zero[:, 1], :2] = (
            (data[non_zero[:, 0], non_zero[:, 1], :2] - center) * factor + center
        )
        return data

    @staticmethod
    def shift(data, x_shift, y_shift):
        non_zero = np.argwhere(np.any(data[:, :, :2] != 0, axis=2))
        data[non_zero[:, 0], non_zero[:, 1], 0] += x_shift
        data[non_zero[:, 0], non_zero[:, 1], 1] += y_shift
        return data

    def reanchor_wrists(self, data):
        left_hand_wrist = data[:, 0, :3]
        right_hand_wrist = data[:, self.HAND_NUM, :3]

        left_pose_wrist = self.HAND_NUM * 2
        right_pose_wrist = self.HAND_NUM * 2 + 1

        data[:, left_pose_wrist, :3] = left_hand_wrist
        data[:, right_pose_wrist, :3] = right_hand_wrist
        return data

    def augment_pipeline(
        self,
        landmarks,
        num=1,
        rotation_axes=None,
        rotation_angles=None,
        zoom_factor=None,
        x_shift=None,
        y_shift=None,
    ):
        normalized_data = landmarks.copy()
        all_versions = [normalized_data]

        for _ in range(num):
            aug_data = normalized_data.copy()
            aug_data = self.reanchor_wrists(aug_data)

            # Rotación
            if rotation_axes and rotation_angles:
                R = self.generate_rotation_matrix_per_axis(rotation_axes, rotation_angles)
                aug_data = self.rotate(aug_data, R)

            # Zoom
            if zoom_factor:
                z_factor = np.random.uniform(*zoom_factor)
                aug_data = self.zoom(aug_data, z_factor)

            # Shift
            if x_shift and y_shift:
                dx = np.random.uniform(*x_shift)
                dy = np.random.uniform(*y_shift)
                aug_data = self.shift(aug_data, dx, dy)

            all_versions.append(aug_data)

        return np.stack(all_versions, axis=0)
