# # For Deployment 

# from prefect.deployments import DeploymentSpec
# from prefect.orion.schemas.schedules import CronSchedule
# from prefect.flow_runners import SubprocessFlowRunner

import pickle
import datetime
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger # add
from prefect.task_runners import SequentialTaskRunner # add

@task #add task
def get_paths(date=None):
    if date==None:
        date = datetime.date.today().replace(day=1)
    else:
        date = datetime.datetime.strptime("-".join(date.split('-')[:-1]), '%Y-%m')
    val_date = (date - datetime.timedelta(days=date.day))
    train_date = (val_date - datetime.timedelta(days=val_date.day))
    v, t = val_date.strftime("%Y-%m"), train_date.strftime("%Y-%m")
    val_path = f"../W1/data/fhv_tripdata_{v}.parquet"
    train_path = f"../W1/data/fhv_tripdata_{t}.parquet"
    return train_path, val_path

@task # add
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task # add
def prepare_features(df, categorical, train=True):
    logger = get_run_logger() # add
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}") #update
    else:
        logger.info(f"The mean duration of validation is {mean_duration}") #update
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task #add
def train_model(df, categorical):
    
    logger = get_run_logger() #add
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}") #update
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features") #update

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}") # update
    return lr, dv

@task #add
def run_model(df, categorical, dv, lr):
    logger = get_run_logger() # add
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}") # update
    return

@flow(task_runner=SequentialTaskRunner()) # add
def main_flow(date=None):
    train_path, val_path = get_paths(date).result()
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result() #update
    
    
    with open(f"./models/dv-{date}.pkl", "wb") as f_out:
        pickle.dump(dv, f_out)
    
    with open(f"./models/model-{date}.bin", "wb") as f_out:
        pickle.dump(lr, f_out)
    run_model(df_val_processed, categorical, dv, lr)

main_flow("2021-08-15")

# DeploymentSpec(    
#     name="Q4-CronScheduleDeployment",
#     flow=main_flow,
#     schedule=CronSchedule(cron="0 9 15 * *"),
#     flow_runner=SubprocessFlowRunner(),
# )
