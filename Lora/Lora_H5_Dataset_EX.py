import torch
from torch.utils.data import Dataset
import h5py
import os


class H5Dataset(Dataset):
    def __init__(self, h5_paths, limit=-1):
        self.limit = limit
        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        self.indices = {}
        idx = 0
        for a, archive in enumerate(self.archives):
            for i in range(len(archive)):
                self.indices[idx] = (a, i)
                idx += 1

        self._archives = None

    @property
    def archives(self):
        if self._archives is None: # lazy loading here!
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    def __getitem__(self, index):
        a, i = self.indices[index]
        archive = self.archives[a]
        dataset = archive[f"trajectory_{i}"]
        data = torch.from_numpy(dataset[:])
        labels = dict(dataset.attrs)

        return {"data": data, "labels": labels}

    def __len__(self):
        if self.limit > 0:
            return min([len(self.indices), self.limit])
        return len(self.indices)


data_path = r"D:\iot\lora\Lora_Records\new_dataset\dataset\Diff_Days"

data_1 = "dataset_220620.h5"
data_2 = "dataset_220627.h5"
data_3 = "dataset_220628.h5"
data_path_1 = os.path.join(data_path, data_1)
data_path_2 = os.path.join(data_path, data_2)
data_path_3 = os.path.join(data_path, data_3)
loader = torch.utils.data.DataLoader(H5Dataset([data_path_1, data_path_2]), num_workers=2)
batch = next(iter(loader))