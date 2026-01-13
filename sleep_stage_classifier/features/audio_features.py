from typing import Tuple
import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import fft


class AudioFeatureExtractor:
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mfcc: int = 40,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        
        self.mel_filterbank = self._create_mel_filterbank()
    
    def _hz_to_mel(self, hz: float) -> float:
        return 2595 * np.log10(1 + hz / 700)
    
    def _mel_to_hz(self, mel: float) -> float:
        return 700 * (10 ** (mel / 2595) - 1)
    
    def _create_mel_filterbank(self) -> np.ndarray:
        low_freq_mel = 0
        high_freq_mel = self._hz_to_mel(self.sample_rate / 2)
        
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.n_mels + 2)
        hz_points = np.array([self._mel_to_hz(m) for m in mel_points])
        
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        filterbank = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        
        for i in range(1, self.n_mels + 1):
            left = bin_points[i - 1]
            center = bin_points[i]
            right = bin_points[i + 1]
            
            for j in range(left, center):
                if center != left:
                    filterbank[i - 1, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right != center:
                    filterbank[i - 1, j] = (right - j) / (right - center)
        
        return filterbank
    
    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        if orig_sr == self.sample_rate:
            return audio
        
        num_samples = int(len(audio) * self.sample_rate / orig_sr)
        resampled = scipy_signal.resample(audio, num_samples)
        return resampled.astype(np.float32)
    
    def normalize(self, audio: np.ndarray) -> np.ndarray:
        audio = audio - np.mean(audio)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        window = scipy_signal.get_window('hann', self.n_fft)
        
        num_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
        spectrogram = np.zeros((self.n_fft // 2 + 1, num_frames))
        
        for i in range(num_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.n_fft] * window
            spectrum = np.abs(fft(frame)[:self.n_fft // 2 + 1])
            spectrogram[:, i] = spectrum
        
        return spectrogram
    
    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        spectrogram = self.compute_spectrogram(audio)
        mel_spec = np.dot(self.mel_filterbank, spectrogram)
        mel_spec = np.log(mel_spec + 1e-10)
        return mel_spec
    
    def compute_mfcc(self, audio: np.ndarray) -> np.ndarray:
        mel_spec = self.compute_mel_spectrogram(audio)
        
        dct_matrix = np.zeros((self.n_mfcc, self.n_mels))
        for k in range(self.n_mfcc):
            for n in range(self.n_mels):
                dct_matrix[k, n] = np.cos(np.pi * k * (2 * n + 1) / (2 * self.n_mels))
        
        mfcc = np.dot(dct_matrix, mel_spec)
        return mfcc
    
    def extract_epoch_features(
        self, 
        audio: np.ndarray, 
        epoch_duration: int = 30
    ) -> Tuple[np.ndarray, np.ndarray]:
        samples_per_epoch = epoch_duration * self.sample_rate
        num_epochs = len(audio) // samples_per_epoch
        
        mel_features = []
        mfcc_features = []
        
        for i in range(num_epochs):
            start = i * samples_per_epoch
            end = start + samples_per_epoch
            epoch_audio = audio[start:end]
            
            epoch_audio = self.normalize(epoch_audio)
            
            mel_spec = self.compute_mel_spectrogram(epoch_audio)
            mfcc = self.compute_mfcc(epoch_audio)
            
            mel_features.append(mel_spec)
            mfcc_features.append(mfcc)
        
        return np.array(mel_features), np.array(mfcc_features)


class AudioFeatureExtractorWithLibrosa:
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mfcc: int = 40,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        
        self._librosa = None
    
    def _get_librosa(self):
        if self._librosa is None:
            import librosa
            self._librosa = librosa
        return self._librosa
    
    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        if orig_sr == self.sample_rate:
            return audio
        librosa = self._get_librosa()
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)
    
    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        librosa = self._get_librosa()
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return librosa.power_to_db(mel_spec, ref=np.max)
    
    def compute_mfcc(self, audio: np.ndarray) -> np.ndarray:
        librosa = self._get_librosa()
        return librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
    
    def extract_epoch_features(
        self, 
        audio: np.ndarray, 
        epoch_duration: int = 30
    ) -> Tuple[np.ndarray, np.ndarray]:
        samples_per_epoch = epoch_duration * self.sample_rate
        num_epochs = len(audio) // samples_per_epoch
        
        mel_features = []
        mfcc_features = []
        
        for i in range(num_epochs):
            start = i * samples_per_epoch
            end = start + samples_per_epoch
            epoch_audio = audio[start:end]
            
            max_val = np.max(np.abs(epoch_audio))
            if max_val > 0:
                epoch_audio = epoch_audio / max_val
            
            mel_spec = self.compute_mel_spectrogram(epoch_audio)
            mfcc = self.compute_mfcc(epoch_audio)
            
            mel_features.append(mel_spec)
            mfcc_features.append(mfcc)
        
        return np.array(mel_features), np.array(mfcc_features)


def get_feature_extractor(use_librosa: bool = True, **kwargs):
    if use_librosa:
        return AudioFeatureExtractorWithLibrosa(**kwargs)
    return AudioFeatureExtractor(**kwargs)
