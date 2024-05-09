from typing import Optional

import pandas as pd
import torch
from torch.fft import fft


class EEGAugmentation:
    def __init__(
            self, eeg_data_df, *,
            time_gaussian_noise_std=0.05,
            spectral_perturbation_mu=0.05,
            spectral_perturbation_eps=0.1,
    ):
        """
        Augments the EEG data for both the time and frequency domain.
        
        Parameters:
            eeg_data_df (pd.DataFrame):
                The input EEG signal tensor.
            time_gaussian_noise_std (float):
                Standard deviation of the Gaussian noise to be added.
            spectral_perturbation_mu (float):
                Probability threshold controlling the perturbation range.
                Used in:
                    - Zero out the amplitudes at locations where U < mu
                    - Replace amplitudes with eps * Am where U > (1 - mu)
            spectral_perturbation_eps (float):
                Scaling factor for adding frequency components.
                Used in:
                    - Replace amplitudes with eps * Am where U > (1 - mu)
        """

        self.eeg_data_df = eeg_data_df
        self.time_gaussian_noise_std = time_gaussian_noise_std
        self.spectral_perturbation_mu = spectral_perturbation_mu
        self.spectral_perturbation_eps = spectral_perturbation_eps

        self.augmented_df: Optional[pd.DataFrame] = None

    def _time_gaussian_noise(self, x):
        """
        Augments the time-domain EEG data by adding Gaussian noise.
        
        Parameters:
        - x (torch.Tensor): The input EEG signal tensor.
        - noise_level (float): Standard deviation of the Gaussian noise to be added.
    
        Returns:
        - torch.Tensor: Augmented EEG signal.
        """
        noise = torch.randn_like(x) * self.time_gaussian_noise_std
        return x + noise

    @staticmethod
    def _freq_fourier_transform(x):
        """
        Applies Fourier transformation to convert time-domain EEG signals into frequency spectra.
        
        Parameters:
        - x (torch.Tensor): The input time-domain EEG signal tensor.
    
        Returns:
        - torch.Tensor: Frequency spectra of the input signals.
        """
        x_fft = fft(x, dim=-1)  # Apply FFT along the last dimension
        return torch.abs(x_fft)  # Return the magnitude spectrum

    def _freq_spectral_perturbation(self, x):
        """
        Perturbs the frequency spectrum weakly by selectively removing and adding frequency components.
        
        Parameters:
        - x (torch.Tensor): The input frequency spectrum tensor.

        Returns:
        - torch.Tensor: Perturbed frequency spectrum.
        """
        # Generate a probability matrix U from a uniform distribution
        U = torch.rand_like(x)

        # Determine maximum amplitude Am from the original frequency spectrum
        Am = torch.max(x)

        # Removing frequency components: zero out the amplitudes at locations where U < mu
        remove_mask = U < self.spectral_perturbation_mu
        x[remove_mask] = 0

        # Adding frequency components: replace amplitudes with epsilon * Am where U > (1 - mu)
        add_mask = U > (1 - self.spectral_perturbation_mu)
        x[add_mask] = self.spectral_perturbation_eps * Am

        return x

    def augment_data(self):
        """ Augments EEG data in both time and frequency domains. """
        augmented_data = []

        for _, row in self.eeg_data_df.iterrows():
            x = torch.tensor(row["EEG"], dtype=torch.float32)

            # Time domain augmentation
            xe = self._time_gaussian_noise(x)

            # Frequency domain augmentation
            xF = self._freq_fourier_transform(x)
            xeF = self._freq_spectral_perturbation(xF)

            augmented_data.append({
                "Subject": row["Subject"],
                "Trial": row["Trial"],
                "Trial_Prefix": row["Trial_Prefix"],
                "Trial_Rep": row["Trial_Rep"],
                "Date": row["Date"],
                "Start_Frame": row["Start_Frame"],

                "EEG": x,  # Original EEG (but in torch format)
                "EEG_Time_Augmented": xe,  # Time Augmented EEG
                "EEG_Frequency": xF,  # Frequency domain representation
                "EEG_Frequency_Augmented": xeF,  # Frequency Augmented EEG
            })

        # Create a new DataFrame with the augmented data
        self.augmented_df = pd.DataFrame(augmented_data)
        return self.augmented_df
