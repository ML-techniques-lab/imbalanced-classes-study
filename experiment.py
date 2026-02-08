"""Imports"""
# Utils
import generate_datasets
import pandas as pd
import random
import sys
import os

# Metrics and model selection
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from imblearn.metrics import geometric_mean_score

# Models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from deslib.des.knora_e import KNORAE
from deslib.dcs import OLA, LCA, MCB
from sklearn_lvq import GlvqModel
from xgboost import XGBClassifier
from deslib.des import KNORAU

"""Constants"""

DATASETS_DIR = "scaled_datasets"
RANDOM_STATE = 10
N_FOLDS = 5
SCALING_METHODS = ['original', 'SS', 'MA', 'RS', 'QT', 'PT', 'MM']
SCORES = {
  'accuracy': accuracy_score,
  'f1_score': f1_score,
  'g_mean': geometric_mean_score,
  'roc_auc': roc_auc_score
}

base_model = Perceptron(random_state=RANDOM_STATE)
pool_classifiers = BaggingClassifier(estimator=base_model, n_estimators=100, random_state=RANDOM_STATE, bootstrap=True,
                                     bootstrap_features=False, max_features=1.0, n_jobs=-1)
MODELS = {
  'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
  'SVM_lin': LinearSVC(random_state=RANDOM_STATE),
  'SVM_rbf': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
  'GLVQ': GlvqModel(prototypes_per_class=1, max_iter=2500, gtol=1e-5, beta=5, random_state=RANDOM_STATE),
  'LR': LogisticRegression(n_jobs=-1, random_state=RANDOM_STATE),
  'GNB': GaussianNB(),
  'GP': GaussianProcessClassifier(1.0 * RBF(1.0), random_state=RANDOM_STATE, n_jobs=-1),
  'LDA': LinearDiscriminantAnalysis(),
  'QDA': QuadraticDiscriminantAnalysis(),
  'DT': DecisionTreeClassifier(random_state=RANDOM_STATE),
  'MLP': MLPClassifier(activation='relu', solver='adam', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=RANDOM_STATE),
  'Percep': Perceptron(random_state=RANDOM_STATE, n_jobs=-1),
  'XGBoost': XGBClassifier(n_jobs=-1, random_state=RANDOM_STATE),
  'RF': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
  'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE),
  'Bagging': pool_classifiers,
  'OLA': OLA(pool_classifiers, random_state=RANDOM_STATE),
  'LCA': LCA(pool_classifiers, random_state=RANDOM_STATE),
  'MCB': MCB(pool_classifiers, random_state=RANDOM_STATE),
  'KNORAE': KNORAE(pool_classifiers, random_state=RANDOM_STATE),
  'KNORAU': KNORAU(pool_classifiers, random_state=RANDOM_STATE)
}

random.seed(RANDOM_STATE)

"""System arguments"""
args = sys.argv

if (args[1] == 'all'):
  scaling_methods = SCALING_METHODS
else:
  scaling_methods = args[1].split(',')
if (args[2] == 'all'):
  models = MODELS.keys()
else:
  models = args[2].split(',')
if len(args) == 5:
  outfile = args[4]
else:
  outfile = 'results.csv'

if (args[3] != 'all'):
  start,end = args[3].split(':')
  start = int(start)
  end = int(end)
else:
  start=1
  end=100

"""Load datasets"""
_, weights = generate_datasets.get_imbalance()
datasets = []
for i in range(start-1, end):
  base_path = f"{DATASETS_DIR}/dataset_{i+1}"
  levels = []
  for j in range(len(weights)):
    weight = weights[j]
    path = f"{base_path}/w_{weight:.3f}"
    csvs = os.listdir(path)
    methods = []
    for k in range(len(scaling_methods)):
      dataset = scaling_methods[k] + ".csv"
      methods.append(pd.read_csv(f"{path}/{dataset}"))
    levels.append(methods)
  datasets.append(levels)


def calculate_score(y_true, y_pred):
  results = [(name, func(y_true, y_pred)) for name, func in SCORES.items()]
  return dict(results)

"""Train models"""
results = {}
folds = StratifiedKFold(n_splits=N_FOLDS, random_state=RANDOM_STATE, shuffle=True)

for name, model in MODELS.items():
  if name not in models:
    continue

  results[name] = []
  
  for i in range(len(datasets)):
    dataset_results = []
    for j in range(len(weights)):
      level_results = []
      for k in range(len(scaling_methods)):
        dataset = datasets[i][j][k]
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]
        model_scores = []
        for train_index, test_index in folds.split(X, y):
          X_train = X.iloc[train_index]
          X_test = X.iloc[test_index]
          y_train = y.iloc[train_index]
          y_test = y.iloc[test_index]
          if (name in ['OLA', 'LCA', 'MCB', 'KNORAE', 'KNORAU']):
            pool_classifiers.fit(X_train, y_train)
          MODELS[name].fit(X_train, y_train)
          model_scores.append(calculate_score(y_test, MODELS[name].predict(X_test)))
        level_results.append(model_scores)
      dataset_results.append(level_results)
    results[name].append(dataset_results)

"""Saving results"""
models_frames = []
for model_name in models:
  datasets_frames = []
  for i in range(len(datasets)):
    weights_frames = []
    for w in range(len(weights)):
      score_frames = [pd.DataFrame(scores, index=[f"fold {i+1}" for i in range(5)]) for scores in results[model_name][i][w]]
      weight_results = pd.concat(score_frames, axis=1, keys=scaling_methods)
      weights_frames.append(weight_results)
    dataset_results = pd.concat(weights_frames, axis=1, keys=[f"level {level+1}" for level in range(len(weights))])
    datasets_frames.append(dataset_results)
  model_results = pd.concat(datasets_frames, axis=1, keys=[f"dataset {d}" for d in range(start, end+1)])
  models_frames.append(model_results)
final_result = pd.concat(models_frames, axis=1, keys=models)
final_result.to_csv(outfile, index=False)
