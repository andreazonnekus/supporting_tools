"""
Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow - 電子書 ISBN: 9781098122461
https://github.com/ageron/handson-ml3/blob/main/01_the_machine_learning_landscape.ipynb
"""

import os, sys, math, pickle, tarfile, urllib
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from joblib import dump, load
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.cluster import ClusterSimilarity

from . import utils, housing_transformer

class HOUSING_HANDSON:
    def __init__(self):
        load_dotenv()
        
        show = False

        self.model_path = os.environ.get("MODEL_PATH")
        np.random.seed(os.environ.get("SEED"))
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
        self.outpath = os.environ.get('OUT_PATH')

        if not self.outpath:
            os.path.join('assets', 'output')
             
    def main(self) -> int:
        if len(sys.argv) > 1:
            img = None
            if len(sys.argv) > 2:
                model = sys.argv[2] if os.path.isfile(os.path.join(self.model_path, sys.argv[2])) else None
                if not model:
                    print('This model doesn\'t exist')

            if len(sys.argv) > 3:
                show = sys.argv[3] if os.path.isfile(os.path.join(self.model_path, sys.argv[3])) else True

            if sys.argv[1] == 'prep':
               x_train, y_train, x_test, y_test = self.prep(self)
            elif sys.argv[1] == 'train':
                x_train, y_train, x_test, y_test = self.prep(self)
                self.train(self, x_train, y_train, x_test, y_test)
            elif sys.argv[1] == 'test':
                self.test(model, img, show)

    def prep(self, dataset = None, show = False, name = ''):
        
        x_train, y_train, x_test, y_test = [], [], [], []

        # if not dataset:
        if not dataset:
            print('Let\'s use the housing data...')
            tarball_path = Path(os.path.join(self.outpath, 'housing.tgz'))
            housing_path = Path(os.path.join(self.outpath, 'housing', 'housing.csv'))
            if not housing_path.is_file():
                Path("datasets").mkdir(parents=True, exist_ok=True)
                url = "https://github.com/ageron/data/raw/main/housing.tgz"
                urllib.request.urlretrieve(url, tarball_path)
                with tarfile.open(tarball_path) as housing_tarball:
                    housing_tarball.extractall(path=self.outpath)

            y_label = 'median_house_value'
            housing = pd.read_csv(housing_path)
            housing['id'] = housing['longitude'] * 1000 + housing['latitude']

            log_pipeline = make_pipeline(
                (SimpleImputer(strategy="median")),
                (FunctionTransformer(np.log, feature_names_out='one-to-one')),
                (StandardScaler()))

            number_pipeline = make_pipeline(
                (SimpleImputer(strategy="median")),
                (StandardScaler()))

            category_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="most_frequent")),
                'categories', OneHotEncoder(handle_unknown = "ignore")
            ])

            cluster_similarity = ClusterSimilarity(n_clusters = 10, gamma = 1., random_state = 42)

            preprocessor = ColumnTransformer([
                ('bedrooms', utils.make_ratio_pipeline(), ['total_bedrooms', 'total_rooms']),
                ('rooms_per_house', utils.make_ratio_pipeline(), ['total_rooms', 'housholds']),
                ('people_per_house', utils.make_ratio_pipeline(), ['population', 'households']),
                ('log', log_pipeline),
                ('geo', cluster_similarity, ['latitude', 'longitude']),
                ('cat', category_pipeline, make_column_selector(dtype_include=object))],
                remainder = number_pipeline)
            
            train_set, test_set = utils.stratified_split(housing, 0.2, 'id')

            # split labels and values
            x_train = train_set.drop(y_label, axis = 1)
            y_train = train_set[y_label].copy()
            x_test = test_set.drop(y_label, axis = 1)
            y_test = test_set[y_label].copy()

            # TODO: save as a CSV output
            x_train.corr().sort_values(ascending = False)
            x_test.corr().sort_values(ascending = False)

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
                
            fig = utils.generate_fig(housing, y_label, c = 'median_house_value', s = 'population')
            utils.save_fig(fig_path, name, fig)
        elif isinstance(dataset, str):
            # TODO: Check if URL or file
            print('Coming soon')
        # else:
            # print('Coming soon')

        return x_train, y_train, x_test, y_test

    def train(self, x_train = None, y_train = None, x_test = None, y_test = None, model_name = None):
        model = LinearRegression()
        is_folder = False

        # TODO: Fix this!
        x_train, y_train, x_test, y_test = self.prep()

        while model_name is None or is_folder is False:
            model_name = input('\nProvide name for an existing model or enter \'new\' to create a new one:\n')
            full_path = os.path.join(os.path.realpath('.'), self.model_path, f'{model_name}')

            if model_name == 'new':
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
                else:
                    model_name = input('\nProvide filename for the new model:\n')

                full_path = os.path.join(os.path.realpath('.'), self.model_path, f'{model_name}')

                if os.path.isfile(full_path):
                    overwrite = input('\nType \'y\' to overwrite the existing file\n')
                    if overwrite == 'y':
                        is_folder = True
                else:
                    is_folder = True

            # housing.to_csv(os.path.join(self.outpath, 'x_train.csv'))
            model.fit(x_train, y_train)

            results = model.score(x_test, y_test)
            # TODO: use this

            dump(model, full_path)
        except Exception as e:
            print(f'Compilation failed: \n\t{e}')
            
        # return model

    def test(self, model = None, name = None, show = False):
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

        # add the data
        x_regg = np.linspace(int(x_min),int(x_max) ,int(x_min/10), np.float64).reshape(-1,1)

        test = input('\nGive a value (float) for the model to predict:\n')
        to_pred = np.array([test], np.float64).reshape(1, 1)
        prediction = model.predict(to_pred)

        figure = utils.add_predictions(fig, x_regg, model.predict(x_regg), to_pred, prediction)

        # save or overwrite
        if overwrite == 'y':
            utils.save_fig(fig_path, name, figure)
        else:
            utils.save_fig(fig_path, new_img, figure)



