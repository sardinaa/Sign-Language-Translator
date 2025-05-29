import cv2
import mediapipe as mp
import numpy as np
import torch
import json

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

filtered_hand = list(range(21))
filtered_pose = [11, 12, 13, 14, 15, 16]
filtered_face = [0, 4, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58,
                 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105,
                 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154,
                 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191,
                 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291,
                 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324,
                 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380,
                 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409,
                 415, 454, 466, 468, 473]

HAND_NUM = len(filtered_hand)
POSE_NUM = len(filtered_pose)
FACE_NUM = len(filtered_face)

# Configura el modelo de MediaPipe
hands = mp.solutions.hands.Hands()
pose = mp.solutions.pose.Pose()
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Carga el vocabulario y el modelo (ajusta las rutas)
with open("vocab_words.json", "r") as f:
    vocab_words = json.load(f)

model = torch.load("results/interpolado_aumentado_20/model.pth").to(device)
model.eval()

def process_landmarks(landmarks_hands, landmarks_pose, landmarks_face):
    """Combina y filtra landmarks como en el entrenamiento."""
    all_landmarks = np.zeros((HAND_NUM*2 + POSE_NUM + FACE_NUM, 2))  # 2D (x, y)
    
    # Manos (derecha e izquierda)
    if landmarks_hands and len(landmarks_hands) >= 1:
        all_landmarks[:HAND_NUM] = np.array([(lm.x, lm.y) for lm in landmarks_hands[0].landmark])[filtered_hand]
    if landmarks_hands and len(landmarks_hands) >= 2:
        all_landmarks[HAND_NUM:HAND_NUM*2] = np.array([(lm.x, lm.y) for lm in landmarks_hands[1].landmark])[filtered_hand]
    
    # Pose
    if landmarks_pose:
        all_landmarks[HAND_NUM*2:HAND_NUM*2 + POSE_NUM] = np.array([(lm.x, lm.y) for lm in landmarks_pose.landmark])[filtered_pose]
    
    # Cara
    if landmarks_face:
        all_landmarks[HAND_NUM*2 + POSE_NUM:] = np.array([(lm.x, lm.y) for lm in landmarks_face.landmark])[filtered_face]
    
    return all_landmarks[:, :2].flatten()  # Aplanar a [360]

# Configura la ventana de tiempo (ej: últimos 30 frames)
sequence_buffer = []
window_size = 30  # Ajusta según el entrenamiento
confidence_threshold = 0.7  # Umbral de confianza para mostrar la palabra

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Extraer landmarks de todas las fuentes
    results_hands = hands.process(frame_rgb)
    results_pose = pose.process(frame_rgb)
    results_face = face_mesh.process(frame_rgb)
    
    if all([results_hands.multi_hand_landmarks, results_pose.pose_landmarks, results_face.multi_face_landmarks]):
        # Procesar y combinar landmarks
        processed_lm = process_landmarks(
            results_hands.multi_hand_landmarks,
            results_pose.pose_landmarks,
            results_face.multi_face_landmarks[0]
        )
        
        # Convertir a tensor y ajustar forma
        input_tensor = torch.tensor(processed_lm).unsqueeze(0).float().to(device)  # Shape: [1, seq_len, input_dim]
        
        # Inferencia
        with torch.no_grad():
            log_probs = model(input_tensor)
            probs = torch.softmax(log_probs, dim=1)  # Convertir logits en probabilidades
            top_prob, top_index = torch.max(probs, dim=1)  # Obtener la predicción más alta
        
        # Agregar a buffer de historial de predicciones
        sequence_buffer.append((top_prob.item(), top_index.item()))
        if len(sequence_buffer) > window_size:
            sequence_buffer.pop(0)

        # Filtrar palabras con confianza suficiente en la ventana de tiempo
        counts = {}
        for prob, idx in sequence_buffer:
            if prob > confidence_threshold:
                if idx in counts:
                    counts[idx].append(prob)
                else:
                    counts[idx] = [prob]

        if counts:
            best_idx = max(counts, key=lambda x: np.mean(counts[x]))
            pred_word = vocab_words[best_idx]
            avg_conf = np.mean(counts[best_idx])
            cv2.putText(frame, f"{pred_word} ({avg_conf:.2f})", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No seguro", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("ASL Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
