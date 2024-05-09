import os
import random
import re
from datetime import datetime
from typing import Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm.auto import tqdm


class SeedDatasetLoader:
    file_pattern = re.compile(r"(\d+)_(\d+)\.mat")

    def __init__(
            self, *,
            preprocessed_eeg_dir="datasets/SEED/Preprocessed_EEG",
            channel_order_filepath="datasets/SEED/channel-order.xlsx",
            seconds_per_eeg=1,
            fs=200,  # Sampling frequency of 200Hz
    ):
        self.preprocessed_eeg_dir = preprocessed_eeg_dir
        self.channel_order_filepath = channel_order_filepath
        self.fs = fs
        self.window_size = seconds_per_eeg * self.fs

        # DataFrame to store EEG data
        self.eeg_data_df = pd.DataFrame()

        # Load channel order
        self.channel_order: Optional[Dict[int, str]] = None
        self._load_channel_order()

        # Load EEG data and labels into DataFrame
        self.labels = None
        self._load_eeg_data()

    def _load_eeg_data(self):
        # Temporary storage list for DataFrame creation
        data_records = []

        # Load all files in the preprocessed_eeg_dir
        for filename in tqdm(os.listdir(self.preprocessed_eeg_dir), desc="Going through files"):
            if filename == "label.mat":
                self.labels = self._load_mat_file(filename)["label"]
            else:
                self._process_file(filename, data_records)

        # Create DataFrame from records
        self.eeg_data_df = pd.DataFrame(data_records, columns=[
            "Subject", "Trial", "Trial_Prefix",
            "Date", "Start_Frame", "EEG",
        ])
        self.eeg_data_df.sort_values(by=["Subject", "Trial", "Date"], inplace=True)

        # Add "Trial_Rep" by counting occurrences of each combination of Subject and Trial
        self.eeg_data_df["Trial_Rep"] = self.eeg_data_df.groupby(["Subject", "Trial"]).cumcount() + 1
        self.eeg_data_df = self.eeg_data_df[[
            "Subject", "Trial", "Trial_Prefix",
            "Trial_Rep",
            "Date", "Start_Frame", "EEG",
        ]]

    def _process_file(self, filename, data_records):
        match = self.file_pattern.match(filename)
        if match:
            subject_nr = int(match.group(1))
            date_str = match.group(2)
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            data = self._load_mat_file(filename)

            for key, value in data.items():
                if "_eeg" in key:
                    prefix, trial_nr = key.split("_eeg")
                    num_segments = value.shape[1] // self.window_size
                    for i in range(num_segments):
                        start_frame = i * self.window_size
                        end_frame = start_frame + self.window_size

                        eeg_segment = value[:, start_frame:end_frame]
                        data_records.append({
                            "Subject": subject_nr,
                            "Trial": int(trial_nr),
                            "Trial_Prefix": prefix,
                            "Date": date_obj,
                            "Start_Frame": start_frame,
                            "EEG": eeg_segment
                        })

    def _load_mat_file(self, filename):
        file_path = os.path.join(self.preprocessed_eeg_dir, filename)
        return loadmat(file_path)

    def _load_channel_order(self):
        self.channel_order = pd.read_excel(self.channel_order_filepath, header=None, index_col=None)[0].to_dict()
        return

    def plot_random_eeg(self):
        # Randomly select an EEG
        random_row = self.eeg_data_df.sample(n=1).iloc[0]
        random_eeg = random_row["EEG"]
        random_channel = random.randint(0, 61)

        # Create a time array based on the number of samples and the sampling rate
        time = np.linspace(0, random_eeg.shape[1] / self.fs, random_eeg.shape[1])

        # Plotting
        plt.figure(figsize=(15, 5), dpi=150)
        plt.plot(time, random_eeg[random_channel, :])
        plt.title(
            f"EEG Channel {random_channel + 1} - "
            f"Subject {random_row['Subject']}, "
            f"Trial {random_row['Trial_Prefix']}, "
            f"Rep {random_row['Trial_Rep']} ({random_row['Date'].strftime('%Y%m%d')}), "
            f"Start_Frame {random_row['Start_Frame']}"
        )
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

    def get_subject_data(self, subject_nr: int, trial_nr: int, trial_rep: int):
        data = self.eeg_data_df[
            (self.eeg_data_df["Subject"] == subject_nr) &
            (self.eeg_data_df["Trial"] == trial_nr) &
            (self.eeg_data_df["Trial_Rep"] == trial_rep)
            ]
        return data.iloc[0]["EEG"] if not data.empty else None

    def get_labels(self) -> np.ndarray:
        return self.labels

    def get_channel_order(self) -> Dict[int, str]:
        return self.channel_order
