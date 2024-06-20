import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.signal import decimate, butter, filtfilt


class EpocDatasetLoader:
    EEG_CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

    def __init__(self, *, my_eeg_dir="datasets/MY_EPOC_X/EEG", seconds_per_eeg=1,
                 original_fs=256, target_fs=200, tolerance=0.01):
        self.my_eeg_dir = my_eeg_dir
        self.seconds_per_eeg = seconds_per_eeg
        self.original_fs = original_fs
        self.target_fs = target_fs
        self.tolerance = tolerance
        self.eeg_cols = [f"EEG.{ch}" for ch in self.EEG_CHANNELS]
        self.window_size = self.seconds_per_eeg * self.original_fs

    def process_dataset(self):
        all_segments = []
        for root, _, files in os.walk(self.my_eeg_dir):
            for file in files:
                if file.endswith(".pm.bp.csv"):
                    eeg_filepath = os.path.join(root, file)
                    json_filepath = eeg_filepath.replace(".pm.bp.csv", ".json")

                    eeg_df = self._load_eeg_data(eeg_filepath)
                    json_data = self._load_json_data(json_filepath)

                    segments = self._extract_eeg_segments(
                        eeg_df,
                        markers=json_data["Markers"],
                        surveys=json_data["surveysResults"]["survey"],
                    )
                    all_segments.extend(segments)

        final_df = pd.DataFrame(all_segments)
        return final_df

    def _extract_eeg_segments(self, eeg_df, markers, surveys):
        eeg_segments = []
        for marker in markers:
            if marker["label"].startswith("video_") and not marker["deleted"]:
                start_time = datetime.fromisoformat(marker["startDatetime"])
                end_time = datetime.fromisoformat(marker["endDatetime"])
                phase_name = marker["extras"]["label_info"]["phase_name"]

                start_timestamp = start_time.timestamp()
                end_timestamp = end_time.timestamp()

                segment_df = eeg_df[(
                        (eeg_df["Timestamp"] >= start_timestamp) &
                        (eeg_df["Timestamp"] <= end_timestamp)
                )]

                if not segment_df.empty:
                    # Ensure all columns exist
                    missing_cols = set(self.eeg_cols) - set(segment_df.columns)
                    if missing_cols:
                        raise Exception(f"Missing columns in EEG data: {missing_cols}")

                    # Check the sample rate accuracy
                    if not self._check_sample_rate(segment_df):
                        raise Exception(f"Sample rate accuracy check failed for phase `{phase_name}`")

                    # Convert DataFrame to NumPy array, transposed to (num_channels, num_samples)
                    eeg_data = segment_df[self.eeg_cols].to_numpy().T
                    num_channels, num_samples = eeg_data.shape
                    if num_channels != len(self.EEG_CHANNELS):
                        raise Exception(f"Unexpected eeg_data shape: {eeg_data.shape}")

                    # Segment the EEG data into 1-second windows
                    num_segments = num_samples // self.window_size
                    for i in range(num_segments):
                        start_frame = i * self.window_size
                        end_frame = start_frame + self.window_size

                        eeg_segment = eeg_data[:, start_frame:end_frame]

                        # Apply bandpass filter
                        eeg_segment = self._apply_bandpass_filter(eeg_segment)

                        # Downsample the data
                        eeg_segment = self._downsample_data(eeg_segment)

                        eeg_tensor = torch.tensor(eeg_segment.copy(), dtype=torch.float32)

                        # Get the verdict from the surveys
                        verdict = next((
                            int(survey["answer_text"][0])
                            for survey in surveys
                            if survey["phase_name"] == phase_name
                        ), None)
                        if verdict is None:
                            raise Exception(f"Answer for phase `{phase_name}` not found!")

                        if "positive" in marker["label"]:
                            verdict = 1
                        elif "neutral" in marker["label"]:
                            verdict = 0
                        elif "negative" in marker["label"]:
                            verdict = -1

                        eeg_segments.append({
                            "Start_Frame": start_frame,
                            "Phase": phase_name,
                            "EEG": eeg_tensor,
                            "Verdict": verdict,
                        })

        return eeg_segments

    def _check_sample_rate(self, segment_df):
        """
        Check if the number of samples in each second is approximately equal to the original_fs.

        Args:
            segment_df (pd.DataFrame): DataFrame containing the segment of EEG data to check.

        Returns:
            bool: True if the sample rate is within the tolerance, False otherwise.
        """
        timestamps = segment_df["Timestamp"].values
        diffs = np.diff(timestamps)
        expected_diff = 1.0 / self.original_fs
        return np.all(np.abs(diffs - expected_diff) < expected_diff * self.tolerance)

    def _downsample_data(self, data):
        """
        Downsample the data to the target_fs using decimation.

        Args:
            data (np.ndarray): Original EEG data. Numpy array of shape (num_channels, num_samples)

        Returns:
            np.ndarray: Downsampled EEG data. Numpy array of shape (num_channels, num_samples_downsampled)
        """
        # Calculate the decimation factor
        decimation_factor = int(self.original_fs / self.target_fs)

        # Apply decimation to each channel
        downsampled_data = decimate(data, decimation_factor, axis=1, zero_phase=True)
        return downsampled_data

    def _apply_bandpass_filter(self, data, low_cut=0.5, high_cut=75.0):
        """
        Apply a bandpass filter to the data.
        Filters frequencies between low_cut and high_cut Hz.

        Args:
            data (np.ndarray): The input data to be filtered. Shape should be (num_channels, num_samples)
            low_cut (float): The lower cutoff frequency in Hz.
            high_cut (float): The upper cutoff frequency in Hz.

        Returns:
            np.ndarray: The bandpass filtered data.
        """
        # TODO: Maybe try with 0.3Hz to 50Hz

        # Calculate the Nyquist frequency
        nyquist = 0.5 * self.target_fs

        # Normalize the cutoff frequencies
        low = low_cut / nyquist
        high = high_cut / nyquist

        # Design a Butterworth bandpass filter
        # noinspection PyTupleAssignmentBalance
        b, a = butter(5, [low, high], btype="band")

        # Apply the filter to the data
        filtered_data = filtfilt(b, a, data, axis=1)
        return filtered_data

    def _load_eeg_data(self, filepath):
        """
        Load EEG data from a CSV file, skipping the first line and keeping only relevant columns.

        Args:
            filepath (str): Path to the CSV file.

        Returns:
            pd.DataFrame: DataFrame containing the EEG data.
        """
        df = pd.read_csv(filepath, skiprows=1)
        columns_to_keep = ["Timestamp"] + self.eeg_cols + ["EQ.OVERALL"]
        missing_cols = set(columns_to_keep) - set(df.columns)
        if missing_cols:
            raise Exception(f"Missing columns in CSV file `{filepath}`: {missing_cols}")
        df = df[columns_to_keep]
        return df

    @staticmethod
    def _load_json_data(filepath):
        """
        Load marker data from a JSON file.

        Args:
            filepath (str): Path to the JSON file.

        Returns:
            dict: Dictionary containing the JSON data.
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        return data
