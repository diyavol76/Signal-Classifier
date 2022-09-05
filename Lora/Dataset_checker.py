import numpy as np
import pandas as pd
import os
import csv

csv_path=r"D:\iot\lora\Lora_Records\new_dataset\concatenated\concatenated.csv"
file_path=r"D:\iot\lora\Lora_Records\new_dataset\concatenated"
SAMPLE_RATE=1000000

df=pd.read_csv(csv_path)

mac_path= df["transmitter_address"]
shape=df.shape
print(shape[0])
size_df=[]
for i in range (df.shape[0]):
    filename = df["Burst_name"][i]
    mac=df["transmitter_address"][i]
    #print(filename)
    #print(mac)
    path=os.path.join(file_path, mac, filename + ".bin")
    #print(path)

    raw_signal = np.fromfile(path, dtype="int16")
    size=len(raw_signal)
    size_df.append(size)

print(size_df)
df["data_size"]=size_df
print(df)
#os.makedirs("D:\iot\lora\Lora_Records\new_dataset\concatenated",exist_ok=True)
pd.DataFrame(df).to_csv(r'D:\iot\lora\Lora_Records\new_dataset\concatenated\with_size_out.csv')
#df.to_csv(r"D:\iot\lora\Lora_Records\new_dataset\concatenated\"with_size_out.csv")