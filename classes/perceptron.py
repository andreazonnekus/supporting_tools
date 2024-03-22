
import random, pickle, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.datasets import make_classification

from utils import *

#We are defining the class of our model
class Perceptron:
    def __init__(self, num_iterations = 1000):
        self.num_iterations = num_iterations
        self.bias = random.uniform(0, 1)
        self.weights = np.array([[ self.bias, self.bias ]])
        self.learning_rate = 0.1

        self.model_path = os.environ.get("MODEL_PATH")
        if not self.model_path:
            print('Please define a MODEL_PATH variable in the .env file\nExiting...')
            exit()
        else:
            if not os.path.exists(self.model_path):
                print('The model path doesn\'t exist. Trying to create it...')
                try:
                    os.makedirs(self.model_path, exist_ok = True)
                except Exception as e:
                    print(e)
                    exit()

    def prep(self, dataset = None, show = False, name = '', num_features = 2):
        n_samples = 100
        x_label = 'feature 1'
        y_label = 'feature 2'
        X, y = make_classification(n_samples=n_samples, n_features=num_features, n_classes=2, n_clusters_per_class=1, n_redundant=0, n_informative=1, random_state=42)

        fig_path = os.path.join('assets', 'output')
        is_file = os.path.isfile(os.path.join(fig_path, name))
        while name is None or is_file is False:
            print(f'\nThese are the existing files:\n\t{os.listdir(os.path.join("assets", "output"))}')
            name = input('\nProvide a name for the figure, you can include the extension or it will default to \'png\':\n')
            
            if len(name.split('.')) > 1:
                is_file = os.path.isfile(os.path.join(fig_path, name))
            else:
                is_file = os.path.isfile(os.path.join(fig_path, f'{name}.png'))
            if is_file:
                    overwrite = input('\nType \'y\' to overwrite the existing file\n')
                    if overwrite == 'y':
                        is_file = True
            else:
                is_file = True
            
        fig = generate_fig(pd.DataFrame(X, columns=[x_label, y_label]), x_label, y_label)
        save_fig(fig_path, name, fig)

        #Using label classes as -1 and 1 to work with our current algorithm
        y[y==0] = -1

        return X, y
    
    def predict(self, x):
        # just apply it here
        return 1 if ((sum(self.weights[-1] * x) + self.bias) > 0) else -1

    def train(self, model_name = None, X = None, y = None):
        is_file = False
        if not X:
            X, y = self.prep()
        
        while model_name is None or is_folder is False:
            model_name = input('\nProvide name for an existing model or enter \'new\' to create a new one:\n')
            full_path = os.path.join(os.path.realpath('.'), self.model_path, f'{model_name}')

            if model_name == 'new':
                # regression with softmax
                is_folder = True
            elif os.path.isfile(full_path):
                self = load(full_path)
                is_folder = True

        is_folder = False
        try:
            while model_name == 'new' or is_folder is False:
                if model_name != 'new':
                    overwrite = input('\nType \'y\' to use overwrite the existing model or type \'n\' to save another:\n')
                    if overwrite == 'y':
                        is_folder = True
                else:
                    model_name = input('\nProvide filename for the new model:\n')

                full_path = os.path.join(os.path.realpath('.'), self.model_path, f'{model_name}')

                if os.path.isfile(full_path):
                    overwrite = input('\nType \'y\' to overwrite the existing file\n')
                    if overwrite == 'y':
                        is_folder = True
                else:
                    is_folder = True

            for _ in range(self.num_iterations):
                index = np.random.choice(len(X))
                x = X[index]
                
                prediction = self.predict(x)
                if(prediction!=y[index]):
                    self.bias = self.learning_rate * (y[index] - prediction) 
                    self.weights = np.append(self.weights, np.expand_dims((self.weights[-1] + self.learning_rate * (y[index] - prediction) * x), axis=0), axis=0)
            dump(self, full_path)
        except Exception as e:
            print(f'Compilation failed: \n\t{e}')

    def test(self, model = None, name = None, x = None):
        new_img, overwrite = '', ''
        x_min, x_max = 0, 0

        if model:
            full_path = os.path.join(os.path.realpath('.'), self.model_path, model)
        
        is_file = os.path.isfile(full_path) if model else False
        while model is None or is_file is False:
            print(f'\nThe following models are available:\n\t{os.listdir(self.model_path)}')
            model = input('\nProvide filename for an existing model or press \'n\' to exit:\n')
            full_path = os.path.join(os.path.realpath('.'), self.model_path, model)

            if model == 'n':
                exit()
            elif os.path.isfile(full_path):
                is_file = True
        model = load(full_path)

        fig_path = os.path.join('assets', 'output')
        print(f'\nThese are the existing files:\n\t{os.listdir(fig_path)}')
        is_file = os.path.isfile(os.path.join(fig_path, name)) if name else False
        while name is None or is_file is False:
            name = input('\nProvide the name for the figure\'s base image and include the extension:\n')

            if os.path.isfile(os.path.join(fig_path, name)):
                    overwrite = input('\nType \'y\' to overwrite the existing file or supply the name of a new image:\n')
                    if overwrite == 'y':
                        is_file = True
                    else:
                        new_img = overwrite
                        print(name, new_img)
                        is_file = True

        # load existing fig        
        with open(os.path.join(fig_path, f'{name.split(".")[0]}.pickle'), 'rb') as file:
            fig = pickle.load(file)
            x_min, x_max = fig.axes[0].get_xlim()
            y_min, y_max = fig.axes[0].get_ylim()
            amount = len(fig.axes[0].get_xdata())

        # add the data
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, amount),
                             np.linspace(y_min, y_max, amount))
        
        test = input('\nGive a value (float) for the model to predict:\n')
        to_pred = np.array([test], np.float64).reshape(1, 1)
        y_line = -(self.weights[:,0][-1] * xx + self.bias) / self.weights[:,1][-1]
        prediction = -(self.weights[:,0][-1] * to_pred + self.bias) / self.weights[:,1][-1]

        figure = add_predictions(fig, xx, y_line, to_pred, prediction)

        # save or overwrite
        if overwrite == 'y':
            save_fig(fig_path, name, figure)
        else:
            save_fig(fig_path, new_img, figure)

per = Perceptron()

per.train()

per.test()