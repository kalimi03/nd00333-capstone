from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory


def clean_data(data):
    # Dict for cleaning data
    regions = {"southwest":0, "southwest":1, "northwest":2, "northeast":3}
    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    charges = pd.get_dummies(x_df.charges, prefix="charges")
    x_df.drop("charges", inplace=True, axis=1)
    x_df = x_df.join(jobs)
    x_df["sex"] = x_df.sex.apply(lambda s: 1 if s == "female" else 0)
    x_df["smoker"] = x_df.smoker.apply(lambda s: 1 if s == "yes" else 0)
    x_df["region"] = x_df.region.map(regions)
    y_df = x_df.pop("charges")
    return x_df, y_df 

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    args = parser.parse_args()
    run = Run.get_context()
    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    wurl='https://raw.githubusercontent.com/kalimi03/nd00333-capstone/master/insurance.csv'
    ds = TabularDatasetFactory.from_delimited_files(wurl)
    x, y = clean_data(ds)
    # Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=42)
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    rms = model.score(x_test, y_test)
    run.log("normalized_root_mean_squared_error", np.float(rms))
    os.makedirs('outputs',exist_ok=True)
    joblib.dump(model,'outputs/model.pkl')
    
if __name__ == '__main__':
    main()