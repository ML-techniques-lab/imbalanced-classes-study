import os

models = [
  ('KNN', 1),
  ('SVM_lin', 1),
  ('SVM_rbf', 1),
  ('GLVQ', 5),
  ('LR', 1),
  ('GNB', 1),
  ('GP', 17),
  ('LDA', 1),
  ('QDA', 1),
  ('DT', 1),
  ('MLP', 3),
  ('Percep', 1),
  ('XGBoost', 2),
  ('RF', 3),
  ('AdaBoost', 9),
  ('Bagging,OLA,LCA,MCB,KNORAE,KNORAU', 17)
]

for model, division in models:
  command = f'sbatch --account=def-menelau --mem=8000M --time=01:30:00 --job-name={model} --array=1-{division} array_job.sh {model}'
  print(f"{model} job submitted")
  os.system(command)