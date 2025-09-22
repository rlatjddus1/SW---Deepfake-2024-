import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class MyTrainDataset(Dataset):
    def __init__(self, file_path, audio_dir):
        self.file_path = file_path
        self.audio_dir = audio_dir
        self.data = pd.read_csv(file_path)

        # Assuming id column is the first column (index 0)
        self.train_file_names = self.data.iloc[:, 0].to_numpy()  # Assuming file names are in the first column

        # Assuming real and fake labels are in the third and fourth columns (index 2 and 3)
        self.real_labels = self.data.iloc[:, 2].to_numpy()  # 'real' column
        self.fake_labels = self.data.iloc[:, 3].to_numpy()  # 'fake' column

        self.num_rows = len(self.data)

        # Convert labels to float32
        self.labels = np.stack((self.real_labels, self.fake_labels), axis=1).astype(np.float32)

    def load_audio(self, file_name):
        audio_path = f"{self.audio_dir}/{file_name}.npy"
        audio_array = np.load(audio_path)
        return audio_array

    def __len__(self):
        return self.num_rows
    
    def __getitem__(self, idx):
        # Get file name and label at specific index
        audio_file = self.train_file_names[idx]
        target = self.labels[idx]

        # Load audio data
        audio = self.load_audio(audio_file)

        # Expand audio data to one channel
        x = audio[np.newaxis, :]

        # Convert to PyTorch tensor
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(target, dtype=torch.float32)

        return x_tensor, y_tensor
