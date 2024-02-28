"""
Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow - 電子書 ISBN: 9781098122461
https://github.com/ageron/handson-ml3/blob/main/01_the_machine_learning_landscape.ipynb
"""

import os, sys, math
import numpy as np
import pandas as pd
from joblib import dump, load
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave
from sklearn.linear_model import LinearRegression

from . import utils

class LINEAR_REGRESSION_HANDSON:
    def __init__(self):
        load_dotenv()

        show = False

        global model_path
        model_path = os.environ.get("MODEL_PATH")
        if not model_path:
            print('Please define a MODEL_PATH variable in the .env file\nExiting...')
            exit()
        else:
            if not os.path.exists(model_path):
                print('The model path doesn\'t exist. Trying to create it...')
                try:
                    os.makedirs(model_path, exist_ok = True)
                except Exception as e:
                    print(e)
                    exit()
             
    def main(self) -> int:
        if len(sys.argv) > 1:
            img = None
            if len(sys.argv) > 2:
                model = sys.argv[2] if os.path.isfile(os.path.join(model_path, sys.argv[2])) else None
                if not model:
                    print('This model doesn\'t exist')

            if len(sys.argv) > 3:
                show = sys.argv[3] if os.path.isfile(os.path.join(model_path, sys.argv[3])) else True

            if sys.argv[1] == 'prep':
               x_train, y_train, x_test, y_test = prep(self)
            elif sys.argv[1] == 'train':
                x_train, y_train, x_test, y_test = prep(self)
                train(self, x_train, y_train, x_test, y_test)
            elif sys.argv[1] == 'test':
                test(model, img, show)

    def prep(self, dataset = None, show = False, name = ''):
        x_train, y_train, x_test, y_test = [], [], [], []
        # if not dataset:
        if not isinstance(dataset, str):
            print('Let\'s assume you want to use Lifesat...')
            data_root = 'https://github.com/ageron/data/raw/main/'
            lifesat = pd.read_csv(f'{data_root}lifesat/lifesat.csv')

            split = 0.9

            x_label = lifesat.columns[1]
            y_label = lifesat.columns[2]
            
            x_train = lifesat[[x_label]][:math.floor(len(lifesat)*split)].values
            x_test = lifesat[[x_label]][math.floor(len(lifesat)*split):].values
            y_train = lifesat[[y_label]][:math.floor(len(lifesat)*split)].values
            y_test = lifesat[[y_label]][math.floor(len(lifesat)*split):].values

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
                
            fig = utils.generate_fig(lifesat, x_label, y_label)
            utils.save_fig(fig_path, name, fig)
        elif isinstance(dataset, str):
            # TODO: Check if URL or file
            print('Coming soon')
        # else:
            # print('Coming soon')

        return x_train, y_train, x_test, y_test

    def train(self, x_train = None, y_train = None, x_test = None, y_test = None, model_name = None):
        is_folder = False

        # TODO: Fix this!
        x_train, y_train, x_test, y_test = self.prep()

        while model_name is None or is_folder is False:
            model_name = input('\nProvide folder name for an existing model or enter \'new\' to create a new one:\n')
            full_path = os.path.join(os.path.realpath('.'), model_path, f'{model_name}.keras')

            if model_name == 'new':
                # regression with softmax
                model = LinearRegression()
                is_folder = True
            elif os.path.isfile(full_path):
                model = load(full_path)
                is_folder = True

        is_folder = False
        try:
            while model_name == 'new' or is_folder is False:
                if model_name != 'new':
                    overwrite = input('\nType \'y\' to use overwrite the existing model or type \'n\' to save another:\n')
                    if overwrite == 'y':
                        is_folder = True

                model_name = input('\nProvide filename for the new model:\n')
                full_path = os.path.join(os.path.realpath('.'), model_path, f'{model_name}.keras')

                if os.path.isfile(full_path):
                    overwrite = input('\nType \'y\' to overwrite the existing file\n')
                    if overwrite == 'y':
                        is_folder = True
                else:
                    is_folder = True
            model.fit(x_train, y_train)
            # print(model.summary())

            results = model.score(x_test, y_test)
            print("Trained model: accuracy of {:5.2f}% with a loss of {:5.2f}".format(100 * results[1], results[0]))

            dump(model, full_path)
        except Exception as e:
            print(f'Compilation failed: \n\t{e}')
            
        # return model

    def test(model = None, name = None, show = False):
        new_img, overwrite = '', ''
        full_path = os.path.join(os.path.realpath('.'), model_path, model)
        is_file = os.path.isfile(full_path)
        while model is None or is_file is False:
            print(f'\nThe following models are available:\n\t{os.listdir(model_path)}')
            model_name = input('\nProvide filename for an existing model or press \'n\' to exit:\n')
            full_path = os.path.join(os.path.realpath('.'), model_path, model_name)

            if model_name == 'n':
                exit()
            elif os.path.isfile(full_path):
                is_file = True
        model = load(full_path)

        fig_path = os.path.join('assets', 'output')
        print(f'\nThese are the existing files:\n\t{os.listdir(fig_path)}')
        is_file = os.path.isfile(os.path.join(fig_path, name))

        while name is None or is_file is False:
            name = input('\nProvide a name for the figure\'s base image:\n')

            if os.path.isfile(os.path.join(fig_path, name)):
                    overwrite = input('\nType \'y\' to overwrite the existing file:\n')
                    if overwrite == 'y':
                        is_folder = True
                    else:
                        new_img = input('\nProvide a name for the new image:\n')
        # load existing image        
        figure = imread(os.path.join(fig_path, name))
        plt.imshow(figure)

        # add the data
        x_regg = np.linspace(0,1,100).reshape(-1,1)

        plt.plot(x_regg, model.predict(x), color = 'blue', linewidth = 1)

        test = input('\nGive a value for the model to predict:')
        pred = model.predict(test)
        plt.plot(test, pred, color = 'black', linewidth = 2)

        # save or overwrite
        if overwrite == 'y':
            utils.save_fig(fig_path, name, figure)
        else:
            utils.save_fig(fig_path, new_img, figure)



