from torchaudio.datasets import VCTK_092
from torchaudio.transforms import Resample
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
import numpy as np

class VCTK_Dataset(Dataset):
    def __init__(self, segment_length, n_speakers, start_idx, end_idx, **kwargs):
        super().__init__()
        vctk = VCTK_092(**kwargs)
        assert (start_idx >= 0 and end_idx >= 0 and start_idx < len(vctk) and end_idx < len(vctk))
        self.x = []
        self.speaker_onehot = []
        for i in range(start_idx, end_idx):
            wav, fs, _, speaker_id, _ = vctk[i]
            wav = Resample(orig_freq=fs, new_freq=16000)(wav).squeeze()
            target_size = int(segment_length * np.ceil(wav.shape[-1] / segment_length))
            wav = F.pad(wav, (int(np.floor((target_size-wav.shape[-1]) / 2)), int(np.ceil((target_size-wav.shape[-1]) /2 )))) # Pad zeros right and left
            wav = torch.split(wav, split_size_or_sections=segment_length)
            spectrograms = []
            for segment in wav:
                spectrogram, _ = mel_spectogram( # Configs from speechbrain vocoder model
                    audio=segment,
                    sample_rate=16000,
                    hop_length=256,
                    win_length=1024,
                    n_mels=80,
                    n_fft=1024,
                    f_min=0.0,
                    f_max=8000.0,
                    power=1,
                    normalized=False,
                    min_max_energy_norm=True,
                    norm="slaney",
                    mel_scale="slaney",
                    compression=True
                )
                spectrograms.append(spectrogram.unsqueeze(0))
            self.x.append(torch.stack(spectrograms, dim=0))
            self.speaker_onehot.append(torch.tensor([vctk._speaker_ids.index(speaker_id) for _ in range(len(spectrograms))])) # F.one_hot(torch.tensor(vctk._speaker_ids.index(speaker_id)), num_classes=n_speakers)[None, :].expand(len(spectrograms), -1)
        self.x = torch.cat(self.x, dim=0)
        self.speaker_onehot = torch.cat(self.speaker_onehot, dim=0)
    def __getitem__(self, index):
        return self.x[index, :, :, :], self.speaker_onehot[index]
    
    def __len__(self):
        return len(self.x)