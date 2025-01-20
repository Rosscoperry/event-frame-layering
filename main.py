import pandas as pd
import numpy as np
from tonic import Dataset, transforms
import torch

class MyRecordings(Dataset):
    sensor_size = (
        1280,
        720,
        2,
    )  # the sensor size of the event camera or the number of channels of the silicon cochlear that was used
    ordering = (
        "xyptc"  # the order in which your event channels are provided in your recordings
    ) # convert to class description 

    def __init__(
        self,
        train=True,
        transform=None,
        target_transform=None,
    ):
        super(MyRecordings, self).__init__(
            save_to='./', transform=transform, target_transform=target_transform
        )
        self.train = train

        # replace the strings with your training/testing file locations or pass as an argument
        if train:
            self.filenames = [
                f"data/recording100.npy"
            ]
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        events = np.load(self.filenames[index])

        if self.transform is not None:
            events = self.transform(events)

        return events

    def __len__(self):
        return len(self.filenames)
    

data_csv1 = pd.read_csv("East_5s.csv")
data_csv2 = pd.read_csv("West_5s.csv")

data_csv1.to_numpy(dtype=np.dtype([("x", int), ("y", int), ("p", int), ("t", int)]))
#data_csv2.to_numpy(dtype=np.dtype([("x", int), ("y", int), ("p", int), ("t", int)]))

# append the two dataframes with the addition of a channel column 
data_csv1['c'] = 0
#data_csv2['c'] = 1

#data_csv = pd.concat([data_csv1, data_csv2])




# def csv_to_nparray(data=data_csv, dtype=np.dtype([("x", int), ("y", int), ("p", int), ("t", int)])):

#     data = data_csv.to_numpy()
#     events = np.zeros(len(data), dtype=dtype)
    
#     events["x"] = data[:,0]
#     events["y"] = data[:,1]
#     events["p"] = data[:,2]
#     events["t"] = data[:,3]
#     return events

np.save(f"data/recording100.npy", data_csv1)

dataset = MyRecordings(train=True, transform=transforms.NumpyAsType(int))

print(dataset)

events = dataset[0]

dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)
events = next(iter(dataloader))

print(events)