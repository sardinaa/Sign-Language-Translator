import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from model.transformer import Transformer
from model.trainner import Trainer
from model.early_stopping import EarlyStopping
from model.dataloader import ZarrDataset
from collections import defaultdict

train_path = "data/train.zarr"
output_folder = "output_processed"

def collate_fn(batch):
    """
    batch: lista de tuplas (lm_tensor, emb_tensor, word_str)
    """
    lm_list, word_list = zip(*batch)

    # Padding para landmarks (src)
    lm_padded = pad_sequence(lm_list, batch_first=True, padding_value=0.0)  # (batch_size, seq_len, input_dim)

    # Máscara para secuencias (True para posiciones válidas, False para padding)
    mask = (lm_padded.abs().max(dim=-1).values > 1e-3) 
    # mask = None

    return lm_padded, mask, list(word_list)

val_path = "data/val_data_classification_top100.zarr"

train_data = ZarrDataset(train_path)
dataloader = DataLoader(train_data, batch_size=128, shuffle=True, collate_fn=collate_fn)

val_data = ZarrDataset(val_path)
val_dataloader = DataLoader(val_data, batch_size=128, shuffle=False, collate_fn=collate_fn)

class_counts = defaultdict(int)
vocab_words = []

# Calcular vocabulario único y conteo por clase
for i in range(len(train_data)):
    _, word = train_data[i]
    if word not in vocab_words:
        vocab_words.append(word)
    class_counts[word] += 1

# Convertir el diccionario a una lista ordenada en el mismo orden que vocab_words
class_counts_list = [class_counts[word] for word in vocab_words]

class_weights = 1.0 / torch.tensor(class_counts_list, dtype=torch.float32)
class_weights /= class_weights.sum()  # Normalizar los pesos

model = Transformer(
    input_dim=train_data[0][0].shape[1],  # Landmark dimension
    num_classes=len(vocab_words),
    d_model=1536,
    num_layers=4,
    num_heads=4,
    d_ff=2056,
    dropout=0.5
)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-3)
scheduler = OneCycleLR(
    optimizer,
    max_lr=2e-4,
    total_steps = len(dataloader) * 10,
    epochs=100,
    final_div_factor=5
)

# Loss function
criterion = nn.CrossEntropyLoss(weight=class_weights.to("cuda"), label_smoothing=0.3)
# Start without label smoothing
early_stopping = EarlyStopping(patience=15, delta=0.0001, path="results/" + output_folder + "/model.pth", verbose=True)

# Trainer initialization
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    dataloader=dataloader,
    val_dataloader=val_dataloader,
    early_stopping=early_stopping,
    scheduler=scheduler,
    device="cuda",
    output_json="results/" + output_folder + "/",
    vocab_words=vocab_words
)

trainer.train(max_epochs=1000)
