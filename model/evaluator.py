import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from .transformer import Transformer
import zarr
import faiss
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import json

class ModelEvaluator:
    def __init__(self, model_path, test_dataloader, zarr_path, device="cuda"):
        """
        Inicializa la clase con el modelo guardado y el DataLoader de test.
        Args:
            model_path (str): Ruta al modelo guardado.
            test_dataloader (DataLoader): DataLoader para los datos de prueba.
            device (str): Dispositivo a usar ('cuda' o 'cpu').
        """
        self.device = device
        print(f"Model path: {model_path}")
        self.model = torch.load(model_path, map_location="cuda")
        self.model.eval()
        self.test_dataloader = test_dataloader

        zarr_store = zarr.open_group(zarr_path, mode="r")
        self.words = [str(zarr_store["words"][str(i)][...]) for i in range(len(zarr_store["words"]))]
        self.embeddings = np.array([zarr_store["embeddings"][str(i)][...] for i in range(len(zarr_store["embeddings"]))])
        embedding_dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(embedding_dim)  # Índice para distancias Euclidianas
        self.index.add(self.embeddings)  # Agregar todos los embeddings

    def get_closest_word_indices(self, embeddings):
        """
        Encuentra los índices de las palabras más cercanas usando FAISS.
        Args:
            embeddings (numpy.ndarray): Batch de embeddings a procesar.
        Returns:
            list: Índices de las palabras más cercanas para cada embedding.
        """
        _, indices = self.index.search(embeddings, 1)  # Buscar el vecino más cercano
        return indices.flatten()  # Retornar los índices como una lista


    def evaluate(self, criterion, output_json="evaluation_results.json"):
        y_true = []
        y_pred = []
        pred_texts = []
        true_texts = []
        total_loss = 0.0

        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs, labels)
                loss = criterion(outputs.squeeze(1), labels)
                total_loss += loss.item()

                # Convertir embeddings a etiquetas discretas
                pred_embeddings = outputs.squeeze(1).cpu().numpy()
                true_embeddings = labels.cpu().numpy()

                pred_labels = self.get_closest_word_indices(pred_embeddings)
                true_labels = self.get_closest_word_indices(true_embeddings)

                y_true.extend(true_labels)
                y_pred.extend(pred_labels)

                # Convertir índices a palabras
                pred_texts.extend([" ".join(self.words[idx] for idx in pred_labels)])
                true_texts.extend([" ".join(self.words[idx] for idx in true_labels)])
            

        pred_texts = [pred.split() for pred in pred_texts]  # Tokeniza predicciones
        true_texts = [[ref.split()] for ref in true_texts]  # Tokeniza referencias

        # Calcular métricas estándar
        metrics = self.compute_metrics(y_true, y_pred)
        metrics["loss"] = total_loss / len(self.test_dataloader)

        # Calcular BLEU-4
        metrics["bleu_4"] = self.compute_bleu_1(pred_texts, true_texts)

        # Calcular ROUGE
        rouge_scores = self.compute_rouge_1(pred_texts, true_texts)
        metrics.update(rouge_scores)

        self.save_results_to_json(metrics, file_path=output_json)

        # Mostrar resultados
        #self.display_confusion_matrix(y_true, y_pred)
        self.display_classification_report(y_true, y_pred)

        print(f"BLEU-4: {metrics['bleu_4']:.4f}")
        print(f"ROUGE Scores: {rouge_scores}")

        return metrics

    def plot_metrics(self, labels, predictions, test_loss):
        """
        Genera gráficos de precisión y pérdida.
        Args:
            labels (list): Etiquetas verdaderas.
            predictions (list): Predicciones del modelo.
            test_loss (float): Pérdida calculada en el conjunto de prueba.
        """
        accuracy = np.mean(np.array(labels) == np.array(predictions))

        print(f"Precisión: {accuracy * 100:.2f}%")
        print(f"Pérdida en test: {test_loss:.4f}")

        plt.figure(figsize=(8, 4))
        plt.title("Pérdida en test")
        plt.bar(["Test Loss"], [test_loss], color="blue")
        plt.ylabel("Pérdida")
        plt.show()

    def compute_metrics(self, y_true, y_pred):
        """Calcula precisión, recall y F1-score."""
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        return {"precision": precision, "recall": recall, "f1_score": f1}

    def display_confusion_matrix(self, y_true, y_pred):
        """Muestra la matriz de confusión."""
        top_labels = np.bincount(y_true).argsort()[-10:]  # Etiquetas más frecuentes
        filtered_indices = [i for i, label in enumerate(y_true) if label in top_labels and y_pred[i] in top_labels]
        filtered_y_true = [y_true[i] for i in filtered_indices]
        filtered_y_pred = [y_pred[i] for i in filtered_indices]
        cm = confusion_matrix(filtered_y_true, filtered_y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("Matriz de Confusión")
        plt.show(block=False)

    def display_classification_report(self, y_true, y_pred):
        """Muestra el reporte de clasificación."""
        print("\nReporte de Clasificación:")
        print(classification_report(y_true, y_pred))

    def compute_bleu_1(self, predictions, references):
        """
        Calcula el puntaje BLEU-1 (basado en unigramas).
        Args:
            predictions (list of list of str): Predicciones generadas por el modelo (tokens).
            references (list of list of list of str): Referencias verdaderas (tokens).
        Returns:
            float: BLEU-1 promedio.
        """
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        smooth = SmoothingFunction().method4
        weights = (1.0, 0.0, 0.0, 0.0)  # Solo unigramas
        bleu_scores = []

        for pred, ref in zip(predictions, references):
            # Aquí ref ya es una lista de listas, no vuelvas a envolverla.
            bleu_scores.append(sentence_bleu(ref, pred, weights=weights, smoothing_function=smooth))

        return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

    def compute_rouge_1(self, predictions, references):
        """
        Calcula ROUGE-1 (unigramas) para palabras aisladas.
        Args:
            predictions (list of list of str): Predicciones generadas por el modelo (tokenizadas).
            references (list of list of list of str): Referencias verdaderas (tokenizadas).
        Returns:
            dict: Contiene precision, recall y F1-score para ROUGE-1.
        """
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)
        precision_list, recall_list, f1_list = [], [], []

        for pred, ref in zip(predictions, references):
            # Convertir listas de palabras en cadenas
            pred_text = " ".join(pred)  # Predicción como cadena
            ref_text = " ".join(ref[0])  # Referencia como cadena (primera referencia)

            # Calcular ROUGE-1
            scores = scorer.score(ref_text, pred_text)
            rouge1 = scores["rouge1"]

            # Guardar métricas
            precision_list.append(rouge1.precision)
            recall_list.append(rouge1.recall)
            f1_list.append(rouge1.fmeasure)

        # Retornar promedios de las métricas
        return {
            "precision_rouge": sum(precision_list) / len(precision_list),
            "recall_rouge": sum(recall_list) / len(recall_list),
            "f1_score_rouge": sum(f1_list) / len(f1_list),
        }

    
    def save_results_to_json(self, metrics, file_path="evaluation_results.json"):
        """
        Guarda los resultados de la evaluación en un archivo JSON.
        Args:
            metrics (dict): Diccionario con las métricas calculadas.
            file_path (str): Ruta del archivo JSON donde se guardarán los resultados.
        """
        try:
            with open(file_path, "w") as json_file:
                json.dump(metrics, json_file, indent=4)
            print(f"Resultados guardados en {file_path}")
        except Exception as e:
            print(f"Error al guardar los resultados: {e}")



