from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.unique_verdicts = sorted(dataframe["Verdict"].unique())
        self.verdict_to_index = {int(verdict): idx for idx, verdict in enumerate(self.unique_verdicts)}
        self.index_to_verdict = {idx: int(verdict) for verdict, idx in self.verdict_to_index.items()}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        xT = row["EEG"].clone().detach()
        xT_augmented = row["EEG_Time_Augmented"].clone().detach()
        xF = row["EEG_Frequency"].clone().detach()
        xF_augmented = row["EEG_Frequency_Augmented"].clone().detach()
        y_idx = self.verdict_to_index[row["Verdict"]]

        return xT, xT_augmented, xF, xF_augmented, y_idx

    def get_class_index(self, verdict):
        return self.verdict_to_index[verdict]

    def get_verdict(self, index):
        return self.index_to_verdict[index]
