import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from model.dataloader import ZarrDataset  # Carga el dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

output_folders = [
    "no_procesado",
    "interpolado",
    "aumentado",
    "interpolado_aumentado_5",
    "interpolado_aumentado_10",
    "interpolado_aumentado_20",
    "interpolado_aumentado_normalizado_5",
    "interpolado_normalizado"
]

output_folders = ["interpolado_aumentado_20"]

# -------------------------------
# Preparar el dataset de test
# -------------------------------
test_path = "data/test_data_classification_top100.zarr"
test_data = ZarrDataset(test_path)

for i in range(len(output_folders)):
    print("🔹 Evaluando modelo en test set con el modelo: " + output_folders[i])

    # -------------------------------
    # Cargar el modelo entrenado
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "results/" + output_folders[i] + "/model.pth"
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()  # Cambiar a modo evaluación

    def collate_fn(batch):
        lm_list, word_list = zip(*batch)
        lm_padded = torch.nn.utils.rnn.pad_sequence(lm_list, batch_first=True, padding_value=0.0)
        mask = (lm_padded.abs().max(dim=-1).values > 1e-6)  # Máscara para padding
        return lm_padded, mask, list(word_list)  # `None` porque no usamos embeddings como `tgt`

    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=True, collate_fn=collate_fn)

    vocab_words = []  # word -> list of embeddings
    for j in range(len(test_data)):
        _, word = test_data[j]
        if word not in vocab_words:
            vocab_words.append(word)  # guardas en lista
    # -------------------------------
    # Realizar Predicciones
    # -------------------------------
    all_true_words = []
    all_pred_words = []

    with torch.no_grad():
        for src, mask, words in test_dataloader:
            src = src.to(device)
            mask = mask.to(device)

            pred = model(src, mask)

            # Obtener índice de la palabra predicha
            predicted_indices = torch.argmax(pred, dim=-1)
            predicted_indices = predicted_indices.squeeze()  # Convertir a 1D si es necesario

            # Convertir índices a palabras
            batch_pred_words = [str(vocab_words[idx]) for idx in predicted_indices.flatten().tolist()]
            batch_true_words = list(words)

            # Acumular
            all_pred_words.extend(batch_pred_words)
            all_true_words.extend(batch_true_words)

    # Verificar que las longitudes coinciden
    assert len(all_true_words) == len(all_pred_words), "Error: Mismatch en longitudes de listas"

    # -------------------------------
    # Calcular Métricas de Evaluación
    # -------------------------------
    accuracy = accuracy_score(all_true_words, all_pred_words)
    precision = precision_score(all_true_words, all_pred_words, average="weighted", zero_division=0)
    recall = recall_score(all_true_words, all_pred_words, average="weighted", zero_division=0)
    f1 = f1_score(all_true_words, all_pred_words, average="weighted", zero_division=0)

    print("\n**Resultados en Test Set:**")
    print(f"Test Accuracy:  {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1-score:  {f1:.4f}")

    # -------------------------------
    # Construir Matriz de Confusión
    # -------------------------------
    conf_matrix = confusion_matrix(all_true_words, all_pred_words, labels=vocab_words)
    conf_df = pd.DataFrame(conf_matrix, index=vocab_words, columns=vocab_words)

    # -------------------------------
    # Identificar las 10 palabras más confundidas
    # -------------------------------
    conf_df.values[np.diag_indices_from(conf_df)] = 0  # Eliminar la diagonal principal
    confused_pairs = conf_df.stack().reset_index()
    confused_pairs.columns = ['True Word', 'Predicted Word', 'Count']
    confused_pairs = confused_pairs.sort_values(by='Count', ascending=False)
    top_confusions = confused_pairs.head(5)

    # -------------------------------
    # Visualizar las palabras más confundidas con un heatmap
    # -------------------------------
    plt.figure(figsize=(10, 8))
    heatmap_data = conf_df.loc[top_confusions["True Word"].unique(), top_confusions["Predicted Word"].unique()]
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Blues', linewidths=0.5)
    plt.xlabel("Predicción")
    plt.ylabel("Palabra Real")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    if output_folders[i]:
            plt.savefig("results/heatmap_" + output_folders[i] + ".png", dpi=500)

