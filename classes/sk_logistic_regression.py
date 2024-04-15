from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, log_loss
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd
import random, os
sb.set_theme(style="white")
plt.rcParams['font.size'] = '12'
plt.style.use('bmh')
np.random.seed(5508)
np.seterr(divide='ignore')

# load and scale down
x_train = pd.read_csv(os.path.join("assets", "train", 'FMNIST_training_set.csv'))/255
y_train = pd.read_csv(os.path.join("assets", "train", 'FMNIST_training_set_labels.csv'))
x_test = pd.read_csv(os.path.join("assets", "train", 'FMNIST_test_set.csv'))/255
y_test = pd.read_csv(os.path.join("assets", "train", 'FMNIST_test_set_labels.csv'))

# break this into a binary classification problem: 7 is sneaker, 5 is sandal
# sneaker must be set to 0
y_train = y_train.loc[(y_train['9']==7) | (y_train['9']==5)]
y_train.loc[(y_train['9']==7)] = 0
y_train.loc[(y_train['9']==5)] = 1
x_train = x_train.loc[y_train.index]

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.2, random_state = 5508)

y_test = y_test.loc[(y_test['9']==7) | (y_test['9']==5)]
y_test.loc[(y_test['9']==7)] = 0
y_test.loc[(y_test['9']==5)] = 1
x_test = x_test.loc[y_test.index]

iterations = 10000
lr = 1E-0
C_range = np.logspace(-10, 5, 30)
                                     
results = {'validation': {}, 'train': {}}
cv = LogisticRegressionCV(Cs=[C_range], cv=10, scoring='log_losss').fit(x_train, y_train)

pred = cv.predict(x_train)
results['train']['misclassification'] = sum(1 for true, pred in zip(y_train, pred) if true != pred) / len(y_train) # get avg
results['train']['log-loss'] = cv.score(x_train, y_train)

pred = cv.predict(x_valid)
results['validation']['misclassification'] = sum(1 for true, pred in zip(y_valid, pred) if true != pred) / len(y_valid) # get avg
results['validation']['log-loss'] = cv.score(x_valid, y_valid)


pred = model.predict(x_train)
print('cost: %.3f f1: %.3f' %(model.cost, f1_score(pred, y_train, average='weighted')))
