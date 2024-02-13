"""
http://yann.lecun.com/exdb/mnist/
"""

import os, sys, keras, cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.datasets.mnist as mnist
from dotenv import load_dotenv
from tensorflow.keras import layers
from tensorflow.keras.saving import save_model, load_model

import utils

def main() -> int:
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
    
    if len(sys.argv) > 1:
        img = None
        if len(sys.argv) > 2:
            model = sys.argv[2] if os.path.isfile(os.path.join(model_path, sys.argv[2])) else None
            if not model:
                print('This model doesn\'t exist')

        if len(sys.argv) > 3:
            img = sys.argv[3] if os.path.isfile(os.path.join('assets', 'input', sys.argv[3])) else None
            if not img:
                print('This image doesn\'t exist')

        if len(sys.argv) > 4:
            show = sys.argv[4] if os.path.isfile(os.path.join(model_path, sys.argv[4])) else True

        if sys.argv[1] == 'prep':
            _, _ = prep()
        elif sys.argv[1] == 'train':
            x_train, y_train, x_test, y_test = prep()
            model = train(x_train, y_train, x_test, y_test)
        elif sys.argv[1] == 'test':
            test(model, img, show)
    
    return 0


def prep(dataset = None):
    if not dataset:
        print('Let\'s assume you want to use MNIST...')
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # normalise the images
        x_train = x_train/255
        x_test = x_test/255

        print(x_train[2].shape)

    return x_train, y_train.astype('i4'), x_test, y_test.astype('i4')

def train(x_train, y_train, x_test, y_test, model_name = None):
    is_folder = False

    # Create a callback that saves the model's weights
    # callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_path, 'callbacks'), save_weights_only = True, verbose = 1)

    while model_name is None or is_folder is False:
        model_name = input('\nProvide folder name for an existing model or enter \'new\' to create a new one:\n')
        full_path = os.path.join(os.path.realpath('.'), model_path, f'{model_name}.keras')

        if model_name == 'new':
            # regression with softmax
            model = tf.keras.models.Sequential([
                layers.Input(x_train.shape[1:]),
                layers.Flatten(),
                layers.Dense(10, 'softmax'),
            ])

            print(model.summary())
            is_folder = True
        elif os.path.isfile(full_path):
            # tf, keras wackiness
            with keras.utils.custom_object_scope({'sparse_softmax_cross_entropy' : tf.compat.v1.losses.sparse_softmax_cross_entropy}):
                model = load_model(full_path)
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

        model.compile(optimizer = 'adam', loss = tf.compat.v1.losses.sparse_softmax_cross_entropy, metrics = ['accuracy'])
        model.fit(x_train, y_train, epochs = 15, batch_size = 128, validation_split = 0.2)

        results = model.evaluate(x_test, y_test)
        print("Trained model: accuracy of {:5.2f}% with a loss of {:5.2f}".format(100 * results[1], results[0]))

        save_model(model, full_path)
    except Exception as e:
        print(f'Compilation failed: \n\t{e}')
    
    return model

def test(model = None, name = None, show = False):
    detected_digits = []
    boxes = []
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
    
    with keras.utils.custom_object_scope({'sparse_softmax_cross_entropy' : tf.compat.v1.losses.sparse_softmax_cross_entropy}):
                model = load_model(full_path)
    is_file = os.path.isfile(os.path.join('assets', 'input', name))

    while name is None or is_file is False:
        print(f'\nThe following files are available:\n\t{os.listdir(os.path.join("assets", "input"))}')
        name = input('\nProvide a valid file to analyse:\n')
        
        if os.path.isfile(os.path.join('assets', 'input', name)):
            is_file = True

    filename = os.path.join('assets', 'input', name)

    img = cv2.imread(filename)
    boxes = utils.boxes(img)

    for box in boxes:
        tst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[box[1]:box[3], box[0]:box[2]]

        new_img = utils.prep_input(tst)

        probs = model.predict(new_img)
        prediction = np.argmax(probs, axis = 1)

        detected_digits.append({'label': prediction[0], 'conf': np.max(probs, axis = 1)[0],  'mn': (box[0], box[1]), 'mx': (box[2], box[3])})

        if show:
            print(f'{probs} => {prediction}')
    
    utils.label_img(detected_digits, img, show)

if __name__ == '__main__':
    main()
    