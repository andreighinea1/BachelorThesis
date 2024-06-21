import pandas as pd
import torch

from dataset_processing.gan.eeg_gan import EegGan


class GanManager:
    def __init__(self, dataframe, generator_layers, discriminator_layers, *,
                 batch_size=10, lstm_dim=128, use_full_lstm=False, latent_dim=100,
                 epochs=100, learning_rate_D=2e-4, learning_rate_G=2e-4, beta1=0.5, beta2=0.999,
                 scheduler_patience_D=10, scheduler_patience_G=10, use_label_smoothing=False,
                 model_save_dir="model_params/gan", log_dir="runs/gan", overwrite_training=False, to_train=True):
        self.dataframe = dataframe
        self.generator_layers = generator_layers
        self.discriminator_layers = discriminator_layers
        self.batch_size = batch_size
        self.lstm_dim = lstm_dim
        self.use_full_lstm = use_full_lstm
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.learning_rate_D = learning_rate_D
        self.learning_rate_G = learning_rate_G
        self.beta1 = beta1
        self.beta2 = beta2
        self.scheduler_patience_D = scheduler_patience_D
        self.scheduler_patience_G = scheduler_patience_G
        self.use_label_smoothing = use_label_smoothing
        self.model_save_dir = model_save_dir
        self.log_dir = log_dir
        self.overwrite_training = overwrite_training
        self.to_train = to_train

        self.eeg_separated_channels, self.verdicts = self._preprocess_eeg_data()
        self.eeg_gans = []

    def _preprocess_eeg_data(self):
        eeg_data = self.dataframe["EEG"].values
        verdicts = self.dataframe["Verdict"].values
        num_channels = eeg_data[0].shape[0]

        separated_channels = [[] for _ in range(num_channels)]
        for eeg in eeg_data:
            for ch in range(num_channels):
                separated_channels[ch].append(eeg[ch].unsqueeze(0))
        separated_channels = [torch.stack(channel_data) for channel_data in separated_channels]

        return separated_channels, verdicts

    def initialize_and_train_gans(self):
        for channel_index, channel_data in enumerate(self.eeg_separated_channels):
            model_save_dir = f"{self.model_save_dir}/channel_{channel_index}"
            log_dir = f"{self.log_dir}/channel_{channel_index}"
            eeg_gan = EegGan(
                channel_data,
                generator_layers=self.generator_layers,
                discriminator_layers=self.discriminator_layers,
                use_full_lstm=self.use_full_lstm,
                batch_size=self.batch_size,
                lstm_dim=self.lstm_dim,
                latent_dim=self.latent_dim,
                epochs=self.epochs,
                learning_rate_D=self.learning_rate_D,
                learning_rate_G=self.learning_rate_G,
                beta1=self.beta1,
                beta2=self.beta2,
                scheduler_patience_D=self.scheduler_patience_D,
                scheduler_patience_G=self.scheduler_patience_G,
                use_label_smoothing=self.use_label_smoothing,
                model_save_dir=model_save_dir,
                log_dir=log_dir,
                overwrite_training=self.overwrite_training,
                to_train=self.to_train,
            )
            eeg_gan.train()
            self.eeg_gans.append(eeg_gan)

    def _generate_data_for_all_channels(self):
        augmented_channels = []
        for gan, channel_data in zip(self.eeg_gans, self.eeg_separated_channels):
            augmented_channel_data = gan.augment_data(channel_data)
            augmented_channels.append(augmented_channel_data)
        return augmented_channels

    def create_augmented_dataframe(self):
        augmented_channels = self._generate_data_for_all_channels()

        # Combine the augmented data into the original dataframe format
        augmented_eeg_data = []
        for i in range(len(augmented_channels[0])):
            eeg_sample = torch.stack([augmented_channels[ch][i] for ch in range(len(augmented_channels))])
            augmented_eeg_data.append(eeg_sample)

        augmented_eeg_data = torch.stack(augmented_eeg_data)

        new_records = []
        for i in range(len(augmented_eeg_data)):
            new_records.append({
                "Start_Frame": 0,
                "Phase": "GAN",
                "EEG": augmented_eeg_data[i],
                "Verdict": self.verdicts[i]
            })
        augmented_df = pd.DataFrame(new_records)

        # Concatenate the original dataframe with the augmented dataframe
        combined_df = pd.concat([self.dataframe, augmented_df], ignore_index=True)

        print(f"Original dataset size: {len(self.dataframe)}")
        print(f"Augmented dataset size: {len(augmented_df)}")
        print(f"Combined dataset size: {len(combined_df)}")
        return combined_df
