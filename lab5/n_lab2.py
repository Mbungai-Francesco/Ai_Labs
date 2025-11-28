from datasets import load_dataset,Audio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, Wav2Vec2Model
import numpy as np

# === Charger le dataset sans décodage audio ===
ds = load_dataset("gilkeyio/AudioMNIST")

ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=True))

audio_path = ds["train"][0]["audio"]["path"]  
print("Chemin du fichier audio :", audio_path)

# === Charger manuellement le son avec librosa ===
audio_array, sr = librosa.load(audio_path, sr=16000)
print("Taille audio :", audio_array.shape)
print("Durée (s) :", len(audio_array) / sr)
print("Taux d’échantillonnage :", sr)

# === Visualisation ===
plt.figure(figsize=(10, 3))
librosa.display.waveshow(audio_array, sr=sr)
plt.title("Waveform d’un échantillon AudioMNIST")
plt.show()

plt.figure(figsize=(10, 4))
spec = librosa.feature.melspectrogram(y=audio_array, sr=sr)
librosa.display.specshow(librosa.power_to_db(spec, ref=np.max), sr=sr, x_axis="time", y_axis="mel")
plt.title("Spectrogramme (échelle Mel)")
plt.colorbar(format='%+2.0f dB')
plt.show()

# === Embeddings Wav2Vec2 ===
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.last_hidden_state
print("Taille des embeddings :", embeddings.shape)

emb_mean = embeddings.mean(dim=1).squeeze().numpy()
print("Taille embedding moyen :", emb_mean.shape)
