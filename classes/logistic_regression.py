from sklearn.model_selection import KFold, train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd
import random, os
sb.set_theme(style="white")
plt.rcParams['font.size'] = '12'
plt.style.use('bmh')
np.random.seed(5508)

class LogisticRegressionClassifier():
    def __init__(self, iterations, learning_rate, verbose = False):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.verbose = verbose
        
        # weight, bias
        # make sure these accept floats, and don't just cast them to 0
        self.probability, self.gradient, self.cost = 0, 0, []
        self.theta = [ 0, 0 ]
        # store this as a lmabda function - Equation 4-13. Logistic Regression model estimated probability (vectorized form)
        # self.estimate_probability_singluar = lambda instance: 1/(1+(np.exp(-np.dot(self.theta[0],np.array(instance, dtype=np.float64).reshape(-1,1)) + self.theta[1])))
        self.estimate_probability = lambda instance: 1/(1+(np.exp(-np.dot(self.theta[0], instance.T) + self.theta[1])))
        # self.estimate_probability = lambda instance: 1/(1+(np.exp(-np.dot(instance,self.theta[0]))))
        # calculate the logistic regression cost function - Equation 4-16. Cost function of a single training instance
        # self.predictor = lambda instance: np.expand_dims((self.estimate_probability(instance) >= 0.5).astype(int), axis=1)
        self.predictor = lambda instance: (self.estimate_probability(instance) >= 0.5).astype(int)

    def train(self, X, y):
        instances, feats = X.shape
        # self.gradient = np.zeros(instances, dtype=np.int64)
        self.theta[0] = np.random.random((1, feats))
        self.theta[1] = random.random()

        for epoch in range(0, self.iterations):
            # ix = random.random(instances)
            # x = X[ix]
            # label = y[ix]

            # TODO: calculate the gradient
            # self.gradient = y.T - (np.dot(self.theta[0], X.T) + self.theta[1])
            self.gradient = y.T - self.estimate_probability(X)
            
            # update theta
            self.theta[0] -= self.learning_rate * -np.mean(np.dot(X.T, self.gradient.T)) 
            self.theta[1] -= self.learning_rate * -np.mean(np.sum(self.gradient))

            # estimate probability
            self.probability = self.estimate_probability(X)

            #calculate cost function
            self.cost.append(np.mean(self.probability - y.T))

            # output some training information,
            # if abs(self.gradient - self.cost[-1]) <= 0.0001:
            #     print(f'converged at {epoch}')
            if epoch % 400 == 399 and self.verbose:
                print(f'0:0.3%', self.cost)

lr = 0.01
iterations = 10000
model = LogisticRegressionClassifier(iterations, lr)

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

model.train(x_train, y_train)