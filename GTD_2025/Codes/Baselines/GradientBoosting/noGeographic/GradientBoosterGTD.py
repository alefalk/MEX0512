import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, callback
import time


class GradientBoosterGTD:
    def __init__(self, trainpath, testpath):
        self.label_encoder = LabelEncoder()
        self.train = pd.read_csv(trainpath, encoding='ISO-8859-1')
        self.test = pd.read_csv(testpath, encoding='ISO-8859-1')

    def splitting(self):
        y_train_raw = self.train['gname']
        y_test_raw = self.test['gname']

        # Encode class labels as integers
        y_train = self.label_encoder.fit_transform(y_train_raw)
        y_test = self.label_encoder.transform(y_test_raw)

        X_train = self.train.drop(columns=['gname'])
        X_test = self.test.drop(columns=['gname'])

        return X_train, X_test, y_train, y_test

    def randomizedSearchXGB(self, X_train, y_train):
        param_grid = {
            'n_estimators': [5, 10, 20, 50, 100, 150, 200, 300, 500],
            'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
            'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.3]
        }

        xgb = XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            tree_method='hist',   
            device='cuda',       
            random_state=42
        )

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        rs = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=param_grid,
            scoring='f1_weighted',
            refit=True,
            n_iter=10,
            return_train_score=True,
            cv=kf,
            n_jobs=-1,
            verbose=2
        )

        rs.fit(X_train, y_train)
        return rs.best_params_

    def train_best_params_xgb(self, best_params, X_train, y_train):
        print("Best params: ", best_params)
        times = []

        # Split off a validation set for early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

        class TimingCallback(callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                if epoch == 0:
                    self.start_time = time.time()
                else:
                    end_time = time.time()
                    elapsed = end_time - self.start_time
                    times.append(elapsed)
                    print(f"Iteration {epoch}, Time: {elapsed:.4f} seconds")
                    self.start_time = end_time
                return False 

        xgbc = XGBClassifier(
            **best_params,
            objective='multi:softprob',
            eval_metric='mlogloss',
            tree_method='hist',
            device='cuda',
            random_state=42
        )

        # Add eval_set and early_stopping_rounds
        xgbc.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            callbacks=[TimingCallback()],
            verbose=False  # suppress XGBoost's own printing
        )

        self.xgb_timing = times
        return xgbc


    def make_predictions(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(Xtest)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy, y_pred, y_proba


def main(trainpath, testpath):
    model = GradientBoosterGTD(trainpath, testpath)
    X_train, X_test, y_train, y_test = model.splitting()

    print("Finding optimal hyperparameters...")
    best_params = model.randomizedSearchXGB(X_train, y_train)

    print("Training best XGBoost classifier...")
    best_model = model.train_best_params_xgb(best_params, X_train, y_train)

    print("Making predictions...")
    accuracy, y_pred, y_proba = model.make_predictions(best_model, X_test, y_test)

    return best_model, accuracy, y_pred, y_test, y_proba, model.xgb_timing


if __name__ == "__main__":
    main()
