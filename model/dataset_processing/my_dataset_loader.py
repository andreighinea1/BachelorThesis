import json
import os
from datetime import datetime

import pandas as pd
import torch


class EpocDatasetLoader:
    EEG_CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

    def __init__(self, *, my_eeg_dir="datasets/MY_EPOC_X/EEG", seconds_per_eeg=1, target_fs=200):
        self.my_eeg_dir = my_eeg_dir
        self.seconds_per_eeg = seconds_per_eeg
        self.target_fs = target_fs
        self.eeg_cols = [f"EEG.{ch}" for ch in self.EEG_CHANNELS]

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

                    eeg_data = segment_df[self.eeg_cols].values
                    eeg_tensor = torch.tensor(eeg_data.T, dtype=torch.float32)

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
                        "Phase": phase_name,
                        "EEG": eeg_tensor,
                        "Verdict": verdict,
                    })

        return eeg_segments

    def _load_eeg_data(self, filepath):
        # Load CSV file, skipping the first line and keeping only relevant columns
        df = pd.read_csv(filepath, skiprows=1)
        columns_to_keep = ["Timestamp"] + self.eeg_cols + ["EQ.OVERALL"]
        missing_cols = set(columns_to_keep) - set(df.columns)
        if missing_cols:
            raise Exception(f"Missing columns in CSV file `{filepath}`: {missing_cols}")
        df = df[columns_to_keep]
        return df

    @staticmethod
    def _load_json_data(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        return data
