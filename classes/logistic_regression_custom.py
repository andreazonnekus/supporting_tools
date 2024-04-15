from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import warnings
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd
import random, os

import utils

sb.set_theme(style="white")
plt.rcParams['font.size'] = '12'
plt.style.use('bmh')
np.random.seed(5508)
np.seterr(divide='ignore')
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class LogisticRegressionClassifier():
    def __init__(self, iterations, learning_rate, C = 1.0, thresh = 0.5, verbose = False):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.threshold = thresh
        self.verbose = verbose
        self.C = C


        # store this as a lambda function - Equation 4-13. Logistic Regression model estimated probability (vectorized form)
        self.estimate_probability = lambda instance: 1/(1+(np.exp(-np.dot(instance, self.theta.T))))
        
        # calculate the logistic regression cost function - Equation 4-16. Cost function of a single training instance
        self.predict = lambda instance: (self.estimate_probability(instance) >= self.threshold).astype(int)

        # self.compute_cost = lambda X, y: (2/len(y) * np.sum(X.T.dot(X.dot(self.theta.T)-y), axis=0))[0]
        self.compute_cost = lambda X, y: np.asarray(((-y * np.log(self.estimate_probability(X)) - ((1.0-y) * np.log(1.0-self.estimate_probability(X)))).sum()/len(y)))[0]

    def train(self, X, y):
        instances, feats = X.shape
        
        # refresh calculations
        self.probability, self.cost, self.cost_history, self.accuracy, self.misclassifications = 0, 0, [], [], []

        # reset the gradient, theta and bias
        self.gradient = np.zeros((feats, 1), dtype=np.int64)
        self.theta = np.random.random((1, feats))

        for epoch in range(0, self.iterations):
            
            # calculate the gradient with l2 regulisarion
            self.gradient = (2/instances) * (np.dot(X.T, (self.probability - y)))

            # avoid the intercept 
            reg = ((self.C / instances) * self.theta)[0:,1]
            self.gradient[1:, 0] += reg
            
            # update theta
            self.theta -= self.gradient.T * self.learning_rate
            
            # estimate probability
            self.probability = self.estimate_probability(X)

            #calculate cost function
            self.cost = -(1/instances) * np.sum(y * np.log(self.probability) + (1 - y) * np.log(1 - self.probability))
            # self.cost =  np.mean(np.sum(X * (self.probability - y), 0))

            # store cost function history
            self.cost_history.append(self.cost)
            self.accuracy.append(accuracy_score(self.predict(X), y))
            # output some training information,
            # if len(self.ch) > 2 and abs(self.ch[-2] - self.cost) <= 0.0001 and self.verbose:
            #     print(f'converged at {epoch}')
            if epoch % 400 == 399 and self.verbose:
                pred = self.predict(X)
                print('cost: %.3f f1: %.3f' %(self.cost, f1_score(pred, y, average='weighted')))

# load and scale down
x_train = pd.read_csv(os.path.join("assets", "train", 'FMNIST_training_set.csv'))
y_train = pd.read_csv(os.path.join("assets", "train", 'FMNIST_training_set_labels.csv'))
x_test = pd.read_csv(os.path.join("assets", "train", 'FMNIST_test_set.csv'))
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
lr = [1E-0, 1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6, 1E-7]
model_accuracies = np.zeros((len(lr), iterations))

for i in range(len(lr)):
    model = LogisticRegressionClassifier(iterations, lr[i], False)
    model.train(x_train, y_train)
    model_accuracies[i] = np.array(model.accuracy)

plt.rc('font', size=12)
plt.rc('axes', titlesize =12)
plt.rc('legend', fontsize=10)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
fig, axarr = figsize=(8, 4)
sb.set_theme(font_scale=1.2)
plt.title('Model accuracy depending selected learning rate', y=1, x=.5)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
model_accuracies_pd = pd.DataFrame(np.array(model_accuracies).transpose(), columns=lr)

available_linestyles = ['-', '--', '-.', ':']
plt.ylim(ymin=0.45, ymax=1)

plt.xlabel('Iterations')
plt.ylabel('Accuracy')

# the plot clearly shows the best learning rate is lr =  based on the maximum accurracy achieved
plt.plot(model_accuracies_pd, label=model_accuracies_pd.columns, linestyle=np.random.choice(available_linestyles))
plt.annotate(f'Optimal η: {lr[np.unravel_index(np.argmax(model_accuracies_pd), model_accuracies_pd.shape)[1]]:.3f} ({np.max(model_accuracies_pd):.3f})', (np.unravel_index(np.argmax(model_accuracies_pd), model_accuracies_pd.shape)[0], np.max(model_accuracies_pd)), xytext=(35, 10), textcoords='offset points', arrowprops=dict(color='black', arrowstyle='->'), fontsize=8)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
utils.save_fig(os.path.join('assets', 'output'), 'logit_theta.png', fig)

optim_lr = lr[np.unravel_index(np.argmax(model_accuracies_pd), model_accuracies_pd.shape)[1]]

calculate_cost = lambda y, prob: -(1/len(y)) * np.sum(y * np.log(prob) + (1 - y) * np.log(1 - prob))
C_range = np.array(sorted(np.logspace(-10, 5, 30)))
model_costs = np.zeros((2,30))
model_misclass = np.zeros((2,30))
iterations = 10000

for ix in range(len(C_range)):
    print(f'currently at {ix}')
    model = LogisticRegressionClassifier(iterations, optim_lr, C_range[ix])
    model.train(x_train, y_train)
    
    # noticed some values of C produce NaN for probabilities - so I'm filtering them out
    proba = model.estimate_probability(x_train)
    if not np.isnan(proba).any():
        pred = (proba >= 0.5).astype(int)
        model_misclass[0][ix] = sum(1 for true, pred in zip(y_train, pred) if true != pred) / len(y_train)
        model_costs[0][ix] = calculate_cost(y_train, proba)
    else: 
        model_misclass[0][ix] = np.inf
        model_costs[0][ix] = np.inf
        
    proba = model.estimate_probability(x_valid)
    if not np.isnan(proba).any():
        pred = (proba >= 0.5).astype(int)
        model_misclass[1][ix] = sum(1 for true, pred in zip(y_valid, pred) if true != pred) / len(y_valid)
        model_costs[1][ix] = calculate_cost(y_valid, proba)
    else: 
        model_misclass[1][ix] = np.inf
        model_costs[1][ix] = np.inf

model_costs_pd = pd.DataFrame(np.transpose(model_costs), columns=['train','validation'], index=C_range)
model_misclass_pd = pd.DataFrame(np.transpose(model_misclass), columns=['train','validation'], index=C_range)

plt.rc('font', size=12)
plt.rc('axes', titlesize =12)
plt.rc('legend', fontsize=10)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
sb.set_theme(font_scale=1.2)
fig, axarr = plt.subplots(1, 2, figsize=(8, 4))
plt.title('Cost and accuracy for diferrent values of (4.2, D7)', y=1.1, x=-0.3)
plt.subplots_adjust(wspace=0.5, hspace=0.5)

plt.xlabel('C_range', x=-0.2)
plt.sca(axarr[0])
plt.xlim(xmin=(min(model_costs_pd.index)-1), xmax=max(model_costs_pd.index)+1)
axarr.flat[0].set_ylabel(f'Model cost', size=10, loc='center')
plt.plot(model_costs_pd, label=model_costs_pd.columns)
plt.annotate(f'Optimal C (Validation): ({C_range[np.argmin(model_costs_pd["validation"])]:.3f},{model_costs_pd["validation"].min():.3f})', 
             (C_range[np.argmin(model_costs_pd["validation"])], min(model_costs_pd["validation"])), 
             xytext=(-35, 10), 
             textcoords='offset points', 
             arrowprops=dict(color='black', arrowstyle='->'), 
             fontsize=8)

plt.sca(axarr[1])
plt.xlim(xmin=(min(model_misclass_pd.index)-1), xmax=max(model_misclass_pd.index)+1)
axarr.flat[1].set_ylabel(f'Model misclassification', size=10, loc='center')
plt.plot(model_misclass_pd, label=model_misclass_pd.columns)
plt.annotate(f'Optimal C (Validation): ({C_range[np.argmin(model_misclass_pd["validation"])]:.3f},{min(model_misclass_pd["validation"]):.3f})', 
             (C_range[np.argmin(model_misclass_pd["validation"])], min(model_misclass_pd["validation"])), 
             xytext=(-35, 10), 
             textcoords='offset points', 
             arrowprops=dict(color='black', arrowstyle='->'), 
             fontsize=8)
plt.legend(bbox_to_anchor=(1.7, 0.5))
utils.save_fig(os.path.join('assets', 'output'), 'logit_c.png', fig)

optim_C = C_range[np.argmin(model_misclass['validation'])]


# model = LogisticRegressionClassifier(iterations, 1E-5, C_range[0], verbose = True)
# model.train(x_train, y_train)

# pred = model.predict(x_train)1
# print('cost: %.3f f1: %.3f' %(model.cost, f1_score(pred, y_train, average='weighted')))
