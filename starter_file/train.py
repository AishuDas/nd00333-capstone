from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

data_loc = "https://raw.githubusercontent.com/AishuDas/nd00333-capstone/master/starter_file/high_diamond_ranked_10min.csv"
ds = TabularDatasetFactory.from_delimited_files(data_loc)

run = Run.get_context()
  
x_df = ds.to_pandas_dataframe().dropna()

y_df = x_df.pop("blueWins")

# TODO: Split data into train and test sets.

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=123)

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--penalty', type=str, default='elasticnet')
    parser.add_argument('--l1_ratio', type=float, default=0.05)
    parser.add_argument('--n_jobs', type=int, default=1)

    args = parser.parse_args()

    model = LogisticRegression(C=args.C, max_iter=args.max_iter, penalty=args.penalty, l1_ratio=args.l1_ratio, n_jobs=args.n_jobs).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    run.log("Regularization:", np.float(args.C))
    run.log("Maximum Iterations:", np.int(args.max_iter))
    pickle.dump(model, 'hyperDrive_{}_{}.pkl'.format(args.C,args.max_iter))

if __name__ == '__main__':
    main()