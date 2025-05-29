import zarr
import torch
from torch.utils.data import Dataset

class ZarrDataset(Dataset):
    """
    Dataset que retorna:
      (landmarks, word_str)
    donde 'landmarks' es [frames, input_dim], 
    'word_str' es la palabra real.
    """
    def __init__(self, zarr_path, landmarks_key="landmarks", word_key="words"):
        self.data = zarr.open_group(zarr_path, mode="r")
        self.landmarks = self.data[landmarks_key]
        self.word = self.data[word_key]
        self.keys = sorted(self.landmarks.keys(), key=lambda x: int(x))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        landmarks = self.landmarks[key][...]
        word_str = str(self.word[key][...])

        if landmarks.shape[0] != 3:
            landmarks = landmarks[:,:,:2]

        landmarks_flat = landmarks.reshape(landmarks.shape[0], -1)
        lm_tensor = torch.from_numpy(landmarks_flat).float()

        return lm_tensor, word_str

