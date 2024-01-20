from torchaudio.datasets import VCTK_092
from torchaudio.transforms import Resample
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

def my_collate(batch):
    """
        Special collate function for the dataloaders to deal with
        input data of varying shapes.
    """
    spectogram = [item[0] for item in batch]
    speaker_id = [item[1] for item in batch]
    return [spectogram, speaker_id]

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
            spectrogram, _ = mel_spectogram( # Configs from speechbrain vocoder model
                audio=wav.squeeze(),
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
            self.x.append(spectrogram.unsqueeze(0))
            self.speaker_onehot.append(F.one_hot(torch.tensor(vctk._speaker_ids.index(speaker_id)), num_classes=n_speakers))

    def __getitem__(self, index):
        return self.x[index], self.speaker_onehot[index]
    
    def __len__(self):
        return len(self.x)