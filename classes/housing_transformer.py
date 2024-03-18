from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
 def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
    self.add_bedrooms_per_room = add_bedrooms_per_room
    
 def fit(self, X, y=None):
    return self # nothing else to do
 
 def transform(self, X, y=None, rooms = 0, households = 0, population = 0, bedrooms = 0):
    rooms_per_household = X[:, rooms] / X[:, households]
    population_per_household = X[:, population] / X[:, households]
    if self.add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms] / X[:, rooms]
        return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]