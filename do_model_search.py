import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import xgboost as xgb
import argparse
from functools import partial

counter: int = 0

categorical_vars = ["main_id_locality", "OrderedDay", "OrderedHour", 
                        #"OrderedMonth"
                        ]
cont_vars = [
             'ETA', 'EDA',
             'latitude', 'del_latitude', 'longitude', 'del_longitude',  
             'IsHoliday', 'AbsLatitudeChange', 'AbsLongitudeChange', 'LatitudeFromCenter', 'LongitudeFromCenter', 'del_LatitudeFromCenter', 
             'del_LongitudeFromCenter', 'points', 
             'min_from_center', 'max_from_center'
             ]

space = {
    'n_estimators': hp.choice('n_estimators', np.arange(100, 500, dtype=int)),
    'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
    'max_depth':  hp.choice('max_depth', np.arange(1, 10, dtype=int)),
    'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    'nthread': 6,
    'booster': 'gbtree',
    'tree_method': 'exact',
    'silent': 1,
    'seed': 42
}


def mape(y_true, y_pred):
    return (np.abs(y_true - y_pred) / y_true).mean()


def load_files(train_file, val_file):
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    return train_df, val_df


def prepare_data(train_df, val_df, do_log=True):
    enc = OneHotEncoder()
    train_oh = enc.fit_transform(train_df[categorical_vars])
    val_oh = enc.transform(val_df[categorical_vars])
    X_train = np.hstack([train_df[cont_vars].values, train_oh.toarray()])
    X_val = np.hstack([val_df[cont_vars].values, val_oh.toarray()])
    y_train = train_df["RTA"]
    y_val = val_df["RTA"]
    if do_log:
        y_train = np.log(y_train)
        y_val = np.log(y_val)
    
    with open("enc.pkl", "wb") as f:
        pickle.dump(enc, f)

    return X_train, X_val, y_train, y_val, enc


def score(X_train, X_val, y_train, y_val, params, do_log=True):
    global counter
    xgb_model = xgb.XGBRegressor(random_state=42, **params)
    print("Params:", params, flush=True)

    xgb_model.fit(X_train, y_train)
    pred = xgb_model.predict(X_val)
    if do_log:
        pred = np.exp(pred)
        y_val = np.exp(y_val)
    score = mape(y_val, pred)
    print("Score ({}): ".format(counter), score, "\n", flush=True)
    with open("xgb_model_{}.pkl".format(counter), "wb") as f:
        pickle.dump(xgb_model, f)

    counter += 1
    return {'loss': score, 'status': STATUS_OK}


def optimize(n_trials, X_train, X_val, y_train, y_val, random_state=42):
    score_ = partial(score, X_train, X_val, y_train, y_val)
    best = fmin(score_, space, algo=tpe.suggest, max_evals=n_trials)
    return best

def refit(X_train, y_train, params):
    xgb_model = xgb.XGBRegressor(random_state=42, **params)
    xgb_model.fit(X_train, y_train)
    with open("xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)

    return xgb_model

def predict(test_file, enc, model, do_log=True):
    test_df = pd.read_csv(test_file)
    test_oh = enc.transform(test_df[categorical_vars])    
    X_test = np.hstack([test_df[cont_vars].values, test_oh.toarray()])
    preds = model.predict(X_test)
    if do_log:
        preds = np.exp(preds)

    prediction = pd.DataFrame({"Id": test_df.Id, "Prediction": preds})
    prediction.to_csv("prediction.csv", index=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    parser.add_argument("val_file")
    parser.add_argument("n_trials", type=int)
    parser.add_argument("--test_file")

    args = parser.parse_args()
    train_file = args.train_file
    val_file = args.val_file
    n_trials = args.n_trials

    train, val = load_files(train_file, val_file)
    X_train, X_val, y_train, y_val, enc = prepare_data(train, val)
    best = optimize(n_trials, X_train, X_val, y_train, y_val)
    model = refit(X_train, y_train, best)

    if args.test_file:
        predict(args.test_file, enc, model)
