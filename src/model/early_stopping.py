import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pth', verbose=False):
        """
        Args:
            patience (int): Cuántas épocas esperar después de no mejorar.
            delta (float): Mínima mejora requerida para resetear el contador.
            path (str): Ruta para guardar el mejor modelo.
            verbose (bool): Si se debe imprimir un mensaje cuando se guarda el modelo.
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Guarda el modelo cuando la pérdida de validación mejora."""
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
        torch.save(model, self.path)
        self.val_loss_min = val_loss