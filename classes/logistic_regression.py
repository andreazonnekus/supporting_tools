import numpy as np
import random
from math import log

class LogisticRegressionClassifier:
    def __init__(self, iterations, learning_rate, verbose = False):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.verbose = verbose
        
        # weight, bias
        # make sure these accept floats, and don't just cast them to 0
        self.gradient, self.cost = None, None        
        self.theta = [0, 0]
        self.probability = 0

        # store this as a lmabda function - Equation 4-13. Logistic Regression model estimated probability (vectorized form)
        self.estimate_probability = lambda instance: 1/(1+np.exp(np.dot(-self.theta, np.array(instance, dtype=np.float64).reshape(-1,1))))
        # calculate the logistic regression cost function - Equation 4-16. Cost function of a single training instance
        self.cost_function = lambda prob, label: -(label*np.log(prob) + (1 - label)*np.log(1-prob))
        self.predict = lambda instance: 1 if (self.estimate_probability(instance) >= 0.5) else 0

    def train(self, X, y):
        instances, feats = X.shape
        self.gradient = np.zeros(instances, dtype=np.int64)
        self.theta[1] = random.uniform(0,1)
        self.theta[0] = np.random.random(feats)

        for epoch in range(0, self.iterations):
            index = np.random.choice(len(X))
            inst = X[index]
            prediction = self.estimate_probability(inst)
            label = y[index]

            # TODO: calculate the gradient
            self.gradient = 
            
            # update theta
            self.theta[1] = #TODO: update this?
            self.theta[0] = np.append(self.theta[0], np.expand_dims((self.theta[0][-1] + self.learning_rate * (label - prediction) * inst), axis=0), axis=0)
    
            # compute probability
            self.probability = self.estimate_probability(inst)
            
            #calculate cost function
            self.cost += self.cost_function(self.probability, label) 

            # output some training information, 
            if epoch % 400 == 399 and self.verbose:
                pass            