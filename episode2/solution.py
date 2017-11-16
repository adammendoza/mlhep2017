#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split,cross_val_score
import xgboost
import pickle

parser = argparse.ArgumentParser(description='Preprocess multiple bricks in parallel.')
parser.add_argument('--input_train', required=True, dest="input_train", metavar="FILE", type=str,
                    help="Input file with train tracks.")
parser.add_argument('--input_test', dest="input_test", metavar="FILE", type=str, default="",
                    help="Input file with test tracks.")
parser.add_argument('--output', dest="output", metavar="FILE", type=str, default="",
                    help="Output file with predictions.")
parser.add_argument('--output_model', dest="output_model", metavar="FILE", type=str, default="",
                    help="Output file with model.")
parser.add_argument('--n_estimators', dest="n_estimators", metavar="N", type=int, default=1000,
                    help="Number of estimators.")
parser.add_argument('--max_depth', dest="max_depth", metavar="N", type=int, default=4,
                    help="Max depth.")
parser.add_argument('--learning_rate', dest="learning_rate", metavar="N", type=float, default=0.3,
                    help="Learning rate.")
parser.add_argument('--n_cross_validations', dest="n_cross_validations", metavar="N", type=int, default=5,
                    help="Number of cross-validation folds.")


args = parser.parse_args()

if len(args.input_test) > 0 and len(args.output) == 0:
    raise RuntimeError("Output is not set.")

if len(args.input_test) == 0 and len(args.output) > 0:
    raise RuntimeError("Input test is not set.")

def AddFields(tracks, field_names):
    n_tracks = tracks.shape[0]
    array = np.zeros(n_tracks)
    for name in field_names:
        tracks[name] = pd.Series(array, tracks.index)

def PrepareTracks(tracks, n_slices):
    for prefix in ['Fwd', 'Bwd']:
        for n in range(1,n_slices):
            pr_1 = '{}{}_'.format(prefix, n)
            pr_2 = '{}{}_'.format(prefix, n+1)
            AddFields(tracks, [pr_2 + 'dn'])
            tracks[pr_2 + 'dn'] = tracks[pr_2 + 'n'] - tracks[pr_1 + 'n']

print "Loading training tracks..."
tracks = pd.read_csv(args.input_train, index_col=0, compression='gzip')

n_slices = 3
PrepareTracks(tracks, n_slices)

names = ["dTX", "dTY", "sTX", "sTY", "chi2", "s_chi2"]
features = ["X", "Y", "Z", "TX", "TY", "chi2"]
for prefix in ['Fwd', 'Bwd']:
    for k in range(0, n_slices):
        for name in names:
            full_name = '{}{}_{}'.format(prefix, k+1, name)
            features.append(full_name)
    features.append('{}1_n'.format(prefix))
    for k in range(1, n_slices):
        full_name = '{}{}_dn'.format(prefix, k+1)
        features.append(full_name)

train, test = train_test_split(tracks, random_state=482603)
X = tracks[features]
X_train = train[features]
X_test = test[features]
Y = tracks['signal']
Y_train = train['signal']
Y_test = test['signal']

print "Starting training..."
xgb = xgboost.XGBClassifier(n_jobs = -1, tree_method = 'exact', n_estimators = args.n_estimators,
                            max_depth = args.max_depth,
                            learning_rate = args.learning_rate).fit(X_train, Y_train, verbose=True)
train_score = roc_auc_score(train.signal, xgb.predict_proba(X_train)[:,1])
test_score = roc_auc_score(test.signal, xgb.predict_proba(X_test)[:,1])
print "\ntrain score = {}".format(train_score)
print "test score = {}\n".format(test_score)
print "(train-test)/(1-train) = {}".format((train_score-test_score)/(1-train_score))

indices = xgb.feature_importances_.argsort()
N = len(indices)
for n in range(0, N):
    print "{}\t{}\t{}".format(n+1, features[indices[N-n-1]], xgb.feature_importances_[indices[N-n-1]])

if len(args.output_model) > 0:
	print "Saving model..."
	pickle.dump(xgb, open(args.output_model, "wb"))

if len(args.input_test) > 0:
    print "Loading testing tracks..."
    test_tracks = pd.read_csv(args.input_test, index_col=0, compression='gzip')
    PrepareTracks(test_tracks, n_slices)
    X_final_test = test_tracks[features]
    print "Estimating final predictions..."
    prediction = xgb.predict_proba(X_final_test)[:, 1]

    print "Storing final predictions..."
    baseline = pd.DataFrame(prediction, columns=['Prediction'])
    baseline.index.name = 'Id'
    baseline.to_csv(args.output, header=True, compression='gzip')

if args.n_cross_validations > 0:
    print "Starting cross-validation..."
    xgb_cv = xgboost.XGBClassifier(n_jobs = -1, tree_method = 'exact', n_estimators = args.n_estimators,
                                   max_depth = args.max_depth, learning_rate = args.learning_rate)
    cv_result = cross_val_score(xgb_cv, X, Y, scoring="accuracy", cv=args.n_cross_validations, verbose=10)
    print "Cross-validation results: {}".format(cv_result)
    print "CV mean: {}, CV std: {}, CV std/mean: {}".format(np.mean(cv_result), np.std(cv_result),
                                                            np.std(cv_result) / np.mean(cv_result))
    print "1-CV mean: {}, 1-CV std: {}, 1-CV std/mean: {}".format(np.mean(1 - cv_result), np.std(1 - cv_result),
                                                            np.std(1 - cv_result) / np.mean(1 - cv_result))
