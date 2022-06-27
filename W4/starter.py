#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd

with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)

categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

# year, month = int(sys.argv[1]), int(sys.argv[2])
year, month = 2021, 4
df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet')
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)

print(f"mean predicted duration: {y_pred.mean()}")

df_result = pd.DataFrame(
    {
        "ride_id": df.ride_id, 
        "predictions": y_pred
    }
)

output_file = "Q2.parquet"
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)