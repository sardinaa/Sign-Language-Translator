import torch
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        dataloader,
        val_dataloader,
        early_stopping,
        scheduler=None,
        device="cuda",
        output_json=None,
        vocab_words=None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.early_stopping = early_stopping
        self.scheduler = scheduler
        self.device = device
        self.output_json = output_json

        # Historial
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "train_precision": [],
            "train_recall": [],
            "train_f1": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": []
        }

        self.vocab_words = vocab_words
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab_words)}


    def train_epoch(self):
        self.model.train()
        train_loss = 0.0

        all_true_labels = []
        all_pred_labels = []

        for src, mask, words in self.dataloader:
            src = src.to(self.device)
            targets = torch.tensor([self.word_to_idx[w] for w in words], 
                       device=self.device, dtype=torch.long)
            
            if mask is not None:
                mask = mask.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(src, mask)
            loss = self.criterion(logits, targets)
            
            assert torch.all(targets >= 0) and torch.all(targets < len(self.vocab_words)), "Error: Índices de targets fuera de rango"

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            if self.scheduler is not None and self.scheduler.last_epoch < self.scheduler.total_steps - 1:
                self.scheduler.step()

            predicted_indices = torch.argmax(logits, dim=-1)
            all_true_labels.extend(targets.cpu().numpy())
            all_pred_labels.extend(predicted_indices.cpu().numpy())

            train_loss += loss.item()

        mean_loss = train_loss / len(self.dataloader)

        # Compute classification metrics.
        accuracy = accuracy_score(all_true_labels, all_pred_labels)
        precision = precision_score(all_true_labels, all_pred_labels, average="weighted", zero_division=0)
        recall = recall_score(all_true_labels, all_pred_labels, average="weighted", zero_division=0)
        f1 = f1_score(all_true_labels, all_pred_labels, average="weighted", zero_division=0)


        return mean_loss, accuracy, precision, recall, f1
    
    def validate(self):
        self.model.eval()
        val_loss = 0.0

        # Lists to accumulate the true and predicted words for metric computation.
        all_true_labels = []
        all_pred_labels = []


        with torch.no_grad():
            for src, mask, words in self.val_dataloader:
                # Move the input tensor and mask to the device.
                src = src.to(self.device)
                if mask is not None:
                    mask = mask.to(self.device)

                # Create target labels from the word list.
                targets = torch.tensor([self.word_to_idx[w] for w in words], 
                       device=self.device, dtype=torch.long)

                # Forward pass through the classifier.
                logits = self.model(src, mask)
                loss = self.criterion(logits, targets)
                val_loss += loss.item()

                # Get the predicted indices and convert them back to words.
                predicted_indices = torch.argmax(logits, dim=-1)
                all_true_labels.extend(targets.cpu().numpy())
                all_pred_labels.extend(predicted_indices.cpu().numpy())

        mean_loss = val_loss / len(self.val_dataloader)

        accuracy = accuracy_score(all_true_labels, all_pred_labels)
        precision = precision_score(all_true_labels, all_pred_labels, average="weighted", zero_division=0)
        recall = recall_score(all_true_labels, all_pred_labels, average="weighted", zero_division=0)
        f1 = f1_score(all_true_labels, all_pred_labels, average="weighted", zero_division=0)

        return mean_loss, accuracy, precision, recall, f1


    def train(self, max_epochs=100):
        for epoch in range(max_epochs):
            train_loss, train_acc, train_prec, train_rec, train_f1 = self.train_epoch()
            val_loss, val_acc, val_prec, val_rec, val_f1 = self.validate()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_accuracy"].append(val_acc)
            self.history["train_precision"].append(train_prec)
            self.history["val_precision"].append(val_prec)
            self.history["train_recall"].append(train_rec)
            self.history["val_recall"].append(val_rec)
            self.history["train_f1"].append(train_f1)
            self.history["val_f1"].append(val_f1)

            print(
                f"Epoch {epoch+1}, "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Prec: {train_prec:.4f}, Train F1: {train_f1:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Prec: {val_prec:.4f}, Val F1: {val_f1:.4f}"
            )

            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        # Guardar historia
        if self.output_json:
            with open(self.output_json + "train_history.json", "w") as f:
                json.dump(self.history, f, indent=4)

        self.plot_history()


    def plot_history(self):
        epochs = range(1, len(self.history["train_loss"]) + 1)

        # Plot de pérdida
        plt.figure()
        plt.plot(epochs, self.history["train_loss"], label="Train Loss")
        plt.plot(epochs, self.history["val_loss"], label="Val Loss")
        plt.title("Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        if self.output_json:
            plt.savefig(self.output_json + "loss.png", dpi=500)
        plt.show(block=False)

        # Plot accuracy
        plt.figure()
        plt.plot(epochs, self.history["val_accuracy"], label="Val Accuracy")
        plt.plot(epochs, self.history["train_accuracy"], label="Train Accuracy")
        plt.title("Accuracy over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        if self.output_json:
            plt.savefig(self.output_json + "accuracy.png", dpi=500)
        plt.show(block=False)
