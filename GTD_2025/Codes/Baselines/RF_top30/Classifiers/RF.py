import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
import time






class RF_nog:
    def __init__(self, trainpath, testpath):
        self.train = pd.read_csv(trainpath, encoding='ISO-8859-1')
        self.test = pd.read_csv(testpath, encoding='ISO-8859-1')

    def splitting(self):
        y_train = self.train['gname']
        X_train = self.train.drop(columns=['gname'])
        y_test = self.test['gname']
        X_test = self.test.drop(columns=['gname'])
        return X_train, X_test, y_train, y_test
    
    # Find optimal parameters for data
    def randomizedSearch(self, X_train, y_train):
        param_grid_rfc = {
            'criterion': ["gini", "entropy"],
            'n_estimators': [5, 10, 20, 50, 100, 150, 200, 300, 500],
            'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'max_features': ['sqrt', 'log2']
            }

        rfc = RandomForestClassifier(random_state=42)

        tscv = TimeSeriesSplit(n_splits=5)

        rs_rfc = RandomizedSearchCV(
            estimator=rfc,
            param_distributions=param_grid_rfc,
            scoring='f1_weighted',
            refit=True,
            n_iter=10,
            return_train_score=True,
            cv=tscv,
            n_jobs=-1,
            verbose=1
        )

        gb_train = rs_rfc.fit(X_train, y_train)
        best_gb = rs_rfc.best_estimator_
        best_params = rs_rfc.best_params_
        best_gb_index = rs_rfc.best_index_
        return best_params
    
    def train_best_params_rf(self, best_params, X_train, y_train):
        rfc = RandomForestClassifier(**best_params, random_state=42)    
        rfc.fit(X_train, y_train)
        return rfc

    def make_predictions(self, best_rfc, X_test, y_test):
        y_pred_gbc = best_rfc.predict(X_test)
        y_proba = best_rcf.predict_proba(Xtest)
        accuracy_gbc = accuracy_score(y_test, y_pred_gbc)
        print(f"Accuracy: {accuracy_gbc * 100:.2f}%")
        return accuracy_gbc, y_pred_gbc, y_proba


def main(trainpath, testpath):
    """Main function to initialize and process data."""
    model = RF_nog(trainpath, testpath)

    X_train, X_test, y_train, y_test = model.splitting()

    print("Finding optimal hyperparameters...")
    best_params = model.randomizedSearch(X_train, y_train)

    print("Training best RF classifier...")
    start_time = time.time()
    best_rfc = model.train_best_params_rf(best_params, X_train, y_train)
    end_time = time.time()

    total_time = end_time - start_time

    print("Making predictions...")
    accuracy_gbc, y_pred_gbc, y_proba = model.make_predictions(best_rfc, X_test, y_test)

    return best_rfc, accuracy_gbc, y_pred_gbc, y_test, y_proba, total_time

if __name__ == "__main__":
    main()  
