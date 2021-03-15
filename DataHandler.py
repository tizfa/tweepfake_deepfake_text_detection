import csv
import pandas as pd

class DataHandler:

    def __init__(self):
        pass

    def readCSVData(self, fname, sep=";"):
        df = pd.read_csv(fname, sep=sep)
        return df

    def testSKLearnModel(self, model, tuned_parameters, train_features, valid_features, test_features, train_labels, val_labels, test_labels):
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import classification_report
        from scipy import sparse
        from sklearn.model_selection import PredefinedSplit

        n_train = len(train_labels)
        n_val = len(val_labels)
        ps = PredefinedSplit(test_fold=[-1] * n_train + [0] * n_val)
        clf = GridSearchCV(model, tuned_parameters, cv=ps)
        all_features = sparse.vstack([train_features, valid_features])

        y_all = train_labels + val_labels
        clf.fit(all_features, y_all)

        print("Model tested: "+str(type(model)))
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = test_labels, clf.predict(test_features)
        print(classification_report(y_true, y_pred, digits=3))
        print()

        return y_pred, clf.best_params_





