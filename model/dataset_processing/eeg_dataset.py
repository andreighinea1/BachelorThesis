from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        xT = row['EEG'].clone().detach()
        xT_augmented = row['EEG_Time_Augmented'].clone().detach()
        xF = row['EEG_Frequency'].clone().detach()
        xF_augmented = row['EEG_Frequency_Augmented'].clone().detach()

        return xT, xT_augmented, xF, xF_augmented
