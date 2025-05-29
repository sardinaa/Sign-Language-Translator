import numpy as np
from copy import deepcopy
from scipy.interpolate import CubicSpline

class HandKeypointInterpolator:
    def __init__(self, hand_num, confidence_threshold_factor=0.5, min_points=5):
        """
        Inicializa el interpolador con configuraciones específicas.
        
        Args:
            hand_num (int): Número de landmarks por mano.
            confidence_threshold_factor (float): Factor para calcular el umbral dinámico de confianza.
            min_points (int): Mínimo de keypoints válidos por mano requeridos para interpolar.
        """
        self.HAND_NUM = hand_num
        self.confidence_threshold_factor = confidence_threshold_factor
        self.min_points = min_points

    def compute_confidence(self, frames):
        """
        Calcula la confianza para cada frame basada en puntos válidos de las manos.
        
        Args:
            frames (numpy.ndarray): Array 3D de forma (frames, landmarks, coordinates).

        Returns:
            numpy.ndarray: Array 1D con las puntuaciones de confianza para cada frame.
        """
        hand_landmarks = frames[:, :self.HAND_NUM * 2, :]
        num_valid_keypoints = np.sum(np.any(hand_landmarks != 0, axis=2), axis=1)
        confidence = num_valid_keypoints / (self.HAND_NUM * 2)  # Normalizado [0, 1]
        return confidence

    def has_minimum_keypoints(self, frame):
        """
        Verifica si el frame tiene suficientes puntos válidos para ambas manos.
        
        Args:
            frame (numpy.ndarray): Array 2D de forma (landmarks, coordinates).

        Returns:
            bool: True si ambas manos tienen suficientes puntos válidos, False en caso contrario.
        """
        left_hand = frame[:self.HAND_NUM]
        right_hand = frame[self.HAND_NUM:self.HAND_NUM * 2]
        valid_left = np.sum(np.any(left_hand != 0, axis=1))
        valid_right = np.sum(np.any(right_hand != 0, axis=1))
        
        return valid_left >= self.min_points and valid_right >= self.min_points

    def interpolate_hand_keypoints(self, frames, max_search_range=5):
        """
        Realiza la interpolación bicúbica para los keypoints de las manos en los frames.
        
        Args:
            frames (numpy.ndarray): Array 3D de forma (frames, landmarks, coordinates).
            max_search_range (int): Rango máximo de búsqueda para interpolación.

        Returns:
            numpy.ndarray: Frames con los puntos interpolados.
        """
        frames_interpolated = deepcopy(frames)
        num_frames, _, _ = frames.shape
        confidence = self.compute_confidence(frames)
        dynamic_threshold = np.mean(confidence) * self.confidence_threshold_factor

        for landmark_idx in range(self.HAND_NUM * 2):
            for k in range(num_frames):
                if np.all(frames_interpolated[k, landmark_idx, :] == 0):
                    # Local temporal window search
                    start = max(0, k - max_search_range)
                    end = min(num_frames, k + max_search_range + 1)
                    
                    window_points = []
                    window_values = []
                    
                    # Collect valid points in local window
                    for ki in range(start, end):
                        if (confidence[ki] >= dynamic_threshold and 
                            self.has_minimum_keypoints(frames_interpolated[ki]) and 
                            not np.all(frames_interpolated[ki, landmark_idx, :] == 0)):
                            
                            window_points.append(ki)
                            window_values.append(frames_interpolated[ki, landmark_idx, :])
                    
                    if len(window_points) >= 4:  # Minimum for cubic spline
                        window_points = np.array(window_points)
                        window_values = np.array(window_values)
                        
                        # Ensure temporal ordering
                        sort_idx = np.argsort(window_points)
                        window_points = window_points[sort_idx]
                        window_values = window_values[sort_idx]
                        
                        # Prevent extrapolation
                        if window_points[0] <= k <= window_points[-1]:
                            try:
                                cs_x = CubicSpline(window_points, window_values[:, 0])
                                cs_y = CubicSpline(window_points, window_values[:, 1])
                                cs_z = CubicSpline(window_points, window_values[:, 2])
                                
                                frames_interpolated[k, landmark_idx, :] = [
                                    cs_x(k),
                                    cs_y(k),
                                    cs_z(k)
                                ]
                            except ValueError:
                                pass  # Fallback to original zeros

        return frames_interpolated
