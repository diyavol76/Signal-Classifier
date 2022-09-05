import os
import pandas
import h5py

data_path= r"D:\iot\lora\Lora_Records\new_dataset\dataset\Diff_Days"

data_1="dataset_220620.h5"
new_data_path=os.path.join(data_path,data_1)

f = h5py.File(new_data_path,'r')
print(f['data'])
