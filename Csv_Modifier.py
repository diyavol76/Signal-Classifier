import pandas as pd
import numpy as np
import math

csv_path=r'D:\iot\IOT_TRAIN\Test\burst_infos_102.csv'
csv_path=r'D:\iot\IOT_TRAIN\Concatenated\all_MAC.csv'

#csv=pd.read_csv(csv_path)

def SNR_selective(df,max_value=np.inf,min_value=0):
    df=df[(df['SNRs']<max_value) & (df['SNRs']>min_value)]

    return df

def Time_selective(df,start_date=0,end_date=np.inf,exact_date=0):

    df=df[df['Burst_name'].str.split(pat="_",n=1,expand=True)[1] > '220214']
    #print(df.iloc[5]['Burst_name'].split("_"))
    return df


if __name__ == '__main__':

    csv = pd.read_csv(csv_path)
    print(csv['transmitter_address'].value_counts())


    #csv_m=csv[(csv['SNRs']>10) & (csv['SNRs']<20) ]
    #print(csv_m['transmitter_address'].value_counts())

    new_csv=SNR_selective(csv)
    print(new_csv['transmitter_address'].value_counts())

    #new_csv_2=Time_selective(csv)
    #print(new_csv_2['transmitter_address'].value_counts())