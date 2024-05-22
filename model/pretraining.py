from torch.utils.data import DataLoader

from dataset_processing.eeg_augmentation import EEGAugmentation
from dataset_processing.eeg_dataset import EEGDataset
from dataset_processing.seed_dataset_loader import SeedDatasetLoader
from model.pre_training.do_pre_training import PreTraining

sampling_frequency = 200  # 200 Hz

_loader = SeedDatasetLoader(fs=sampling_frequency)
labels = _loader.get_labels()
channel_order = _loader.get_channel_order()
_eeg_data_df = _loader.get_eeg_data_df()
_loader.plot_random_eeg()
del _loader

_augmentor = EEGAugmentation(_eeg_data_df)
_augmented_df = _augmentor.augment_data()
del _augmentor, _eeg_data_df

# From the paper
pretraining_batch_size = 256

_dataset = EEGDataset(_augmented_df)
data_loader = DataLoader(_dataset, batch_size=pretraining_batch_size, shuffle=True)
del _augmented_df, _dataset

pretraining = PreTraining(
    data_loader=data_loader,
    sampling_frequency=sampling_frequency,
    pretraining_model_save_dir="model_params/pretraining",
    epochs=10,
)
pretraining.train()
