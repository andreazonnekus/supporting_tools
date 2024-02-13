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

def main() -> int:
    load_dotenv()

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
            img = sys.argv[2] if os.path.isfile(os.path.join('assets', 'input', f'{sys.argv[2]}.jpg')) else None
            if not img:
                print('This image doesn\'t exist')

        if sys.argv[1] == 'prep':
            _, _ = prep()
        elif sys.argv[1] == 'train':
            x_train, y_train, x_test, y_test = prep()
            model = train(x_train, y_train, x_test, y_test)
        elif sys.argv[1] == 'test':
            test(None, img)
    
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

def test(model = None, name = None):
    is_file = False
    while model is None or is_file is False:
        print(f'\nThe following models are available:\n\t{os.listdir(model_path)}')
        model_name = input('\nProvide filename for an existing model or press \'n\' to exit:\n')
        full_path = os.path.join(os.path.realpath('.'), model_path, model_name)

        if model_name == 'n':
            exit()
        elif os.path.isfile(full_path):
            with keras.utils.custom_object_scope({'sparse_softmax_cross_entropy' : tf.compat.v1.losses.sparse_softmax_cross_entropy}):
                model = load_model(full_path)
            is_file = True

    is_file = False

    print(f'\nThe following files are available:\n\t{os.listdir(os.path.join("assets", "input"))}')
    while name is None or is_file is False:
        name = input('\nProvide a valid file to analyse:\n')
        
        if os.path.isfile(os.path.join('assets', 'input', name)):
            is_file = True

    filename = os.path.join('assets', 'input', name)
    print(filename)
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    mod_img = img.copy()
    mod_img.astype('uint8', copy = False)

    mod_img = cv2.threshold(mod_img, 146, 255, cv2.THRESH_BINARY_INV)[1]

    mod_img = cv2.resize(mod_img, (20, 20), interpolation = cv2.INTER_NEAREST) # The MNIST preprocessing preserves the image ratio
    print(mod_img)

    new_img = np.full((28, 28), 0, dtype = np.uint8)
    new_img[4:4 + mod_img.shape[0], 4:4 + mod_img.shape[1]] = mod_img
    new_img = np.expand_dims(new_img, 0)
    probs = model.predict(new_img)
    prediction = np.argmax(probs, axis = 1)

    print(f'{probs} => {prediction}')
    cv2.imshow(name, new_img[0])
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
    