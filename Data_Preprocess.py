import RF_Dataset
import pandas as pd
csv_path=r"D:\iot\IOT_TRAIN\Test\burst_infos_102.csv"
file_path=r"D:\iot\IOT_TRAIN\Test"


csv=pd.read_csv(csv_path)
#csv[csv[]]

train_dataset = RF_Dataset.RF_Dataset(csv_path, file_path )
test_dataset = RF_Dataset.RF_Dataset(csv_path, file_path)


image, label = train_dataset[0]
print(label)
print(train_dataset[3])