import os, sys, cv2, math, pickle, nltk, re, wikipedia
from zlib import crc32
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as sw
from nltk.tokenize import word_tokenize
import gensim.downloader as gen_api
import seaborn as sb
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import train_test_split
import torch

load_dotenv()

column_ratio = lambda X: X[:, 0] / X[:, 1]
ratio_name = lambda transformer, feature_names_in: ['ratio']
prepare_vocab_dict = lambda char_arr: {n: i for i, n in enumerate(char_arr)}

def make_ratio_pipeline() -> Pipeline:
    return make_pipeline([
            (SimpleImputer(strategy="median")),
            (FunctionTransformer(column_ratio, feature_names_out = ratio_name)),
            (StandardScaler())
        ])

def box(img, show = False):
    """
    Apply thresholding and rectangle detection on the input image.

    Parameters:
    - img: Input image as a filename (string) or numpy array.
    - show (bool, optional): Whether to display intermediate results. Defaults to False.

    Returns:
    Tuple: Coordinates of the detected rectangle.
    """

    # if a string, assume not an array
    if isinstance(img, str):
        img = cv2.imread(img)
    elif isinstance(img, np.ndarray):
        print('\nIs already array')

    mod_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mod_img.astype('uint8', copy = False)

    thresh = cv2.threshold(mod_img, 146, 255, cv2.THRESH_BINARY)[1]

    if show:
        cv2.imshow('tst', thresh)
        cv2.waitKey(0)

    contour, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contour)
    x_max = x + w
    y_max = y + h

    cv2.rectangle(img, (x,y), (x_max, y_max), (255, 0, 0), 1)

    if show:
        cv2.imshow('tst', img)
        cv2.waitKey(0)
        
        cv2.imwrite('Final Result.jpg',img)

    return (x, y), (x_max, y_max)

def boxes(img, show = False):
    """
    Apply thresholding and detect multiple rectangles in the input image.

    Parameters:
    - img: Input image as a filename (string) or numpy array.
    - show (bool, optional): Whether to display intermediate results. Defaults to False.

    Returns:
    List: List of bounding box coordinates.
    """
    boxes = []

    # if a string, assume not an array
    if isinstance(img, str):
        img = cv2.imread(img)
    elif isinstance(img, np.ndarray) and show:
        print('\nIs already array')

    mod_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mod_img.astype('uint8', copy = False)

    thresh = cv2.threshold(mod_img, 146, 255, cv2.THRESH_BINARY_INV)[1]

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if show:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        boxes.extend([(x, y, x + w, y + h)])
    
    if show:
        cv2.imshow('tst', img)
        cv2.waitKey(0)
        
        cv2.imwrite('Final Result.jpg',img)
    
    return boxes

def prep_input(img, show = False):
    """
    Preprocess input image for further analysis.

    Parameters:
    - img: Input image as a filename (string) or numpy array.
    - show (bool, optional): Whether to display intermediate results. Defaults to False.

    Returns:
    np.ndarray: Preprocessed image array.
    """
    # if a string, assume not an array
    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    elif isinstance(img, np.ndarray) and show:
        print('\nIs already array')

    # more suitable for images with actual complexity
    mod_img = img.copy()
    mod_img.astype('uint8', copy = False)

    # mod_img = cv2.bilateralFilter(src=mod_img, d=9, sigmaColor=9, sigmaSpace=7)
    # if show:
    #     cv2.imshow("Output: ", mod_img)
    #     cv2.waitKey(0)

    # _, mod_img = cv2.threshold(mod_img,127,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)

    mod_img = cv2.threshold(mod_img, 146, 255, cv2.THRESH_BINARY_INV)[1]

    mod_img = cv2.resize(mod_img, (20, 20), interpolation = cv2.INTER_NEAREST) # The MNIST preprocessing preserves the image ratio

    new_img = np.full((28, 28), 0, dtype = np.uint8)

    new_img[4:4 + mod_img.shape[0], 4:4 + mod_img.shape[1]] = mod_img
    if show:
        print(new_img)
        np.savetxt(os.path.join('assets', 'output', 'digit.txt'), new_img, delimiter=',', fmt='%d')
    
    return np.expand_dims(new_img, 0)

def label_img(input_dicts, img, show = False):
    """
    Label bounding boxes on the input image.

    Parameters:
    - input_dicts: List of dictionaries containing bounding box information.
    - img: Input image as a numpy array.
    - show (bool, optional): Whether to display the labeled image. Defaults to False.
    """
    for input_dict in input_dicts:
        if show:
                print(input_dict)
        cv2.rectangle(img, input_dict['mn'], input_dict['mx'], (255, 0, 0), 1)
        cv2.putText(img, str(input_dict['label']), (int(input_dict['mn'][0] +(input_dict['mx'][0] - input_dict['mn'][0]) / 4), input_dict['mx'][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        if show:
            cv2.imshow('final', img)
            cv2.waitKey(0)
        cv2.imwrite(os.path.join('assets', 'output', 'final.jpg'),img)

def generate_fig(data, x_label = None, y_label = None, show = False, 
                 c = None, s = None, height = 6, width = 8, line_width = 1, alpha = 0.4, 
                 style = 'ggplot', cmap = 'jet', subplots = False, show_grid = True, 
                 title_size = 14, label_size = 10):
    """
    Generate a plot based on input data.

    Parameters:
    - data: Input data.
    - x_label (optional): Label for the x-axis. Defaults to None.
    - y_label (optional): Label for the y-axis. Defaults to None.
    - show (bool, optional): Whether to display the plot. Defaults to False.
    - c (optional): Color parameter. Defaults to None.
    - s (optional): Size parameter. Defaults to None.
    - height (int, optional): Height of the plot. Defaults to 6.
    - width (int, optional): Width of the plot. Defaults to 8.
    - line_width (int, optional): Line width. Defaults to 1.
    - alpha (float, optional): Transparency of markers. Defaults to 0.4.
    - style (str, optional): Plot style. Defaults to 'ggplot'.
    - cmap (str, optional): Color map. Defaults to 'jet'.
    - subplots (bool, optional): Whether to use subplots. Defaults to False.
    - show_grid (bool, optional): Whether to display grid lines. Defaults to True.
    - title_size (int, optional): Font size of the title. Defaults to 14.
    - label_size (int, optional): Font size of labels. Defaults to 10.

    Returns:
    plt.Figure: Generated plot figure.
    """
    
    plt.rc('font', size=12)
    plt.rc('axes', labelsize = label_size, titlesize = title_size)
    plt.rc('legend', fontsize=12)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(1, 1, 1)

    # add a param to use different styles etc
    # color_map = 'viridis'
    # color = 'k'
    plt.style.use(style)

    # typically an index column to what is otherwise one feature and one target
    if not x_label and len(data.columns) == 3:
        x_label = data.columns[1]
    if not y_label and len(data.columns) == 3:
        y_label = data.columns[2]
    
    # multiple features
    if y_label and len(data.columns) > 3:
        fig, axarr = plt.subplots(2)
        cmap = sb.diverging_palette(230, 20, as_cmap=True)

        # covariance matrix
        plt.set(axarr[0])
        x = data.drop(y_label, axis = 1)
        cov_mat = np.cov(x.T, bias= True)

        plt.figure(figsize=(13,13))
        sb.set_theme(font_scale=1.2)
        hm = sb.heatmap(cov_mat,
                        cbar=True,
                        annot=True,
                        square=True,
                        fmt='.2f',
                        annot_kws={'size': 12},
                        yticklabels=x.columns,
                        xticklabels=x.columns)
        plt.title('Covariance matrix showing correlation coefficients')

        # pairplot
        if y_label and data[y_label] != 'O':
            color_dict = {data[y_label].min(): 'red', data[y_label].max()*0.25: 'blue', data[y_label].max()*0.5: 'green', data[y_label].max()*0.75: 'orange', data[y_label].max(): 'purple'}
        else:
            color_dict = ['#4575b4', '#91bfdb', '#e0f3f8', '#fee090']

        plt.set(axarr[1])
        sb.pairplot(data, kind='reg', hue=data[y_label], palette=color_dict)
        plt.title(f'Pairplot of ', y=1.2)
        plt.tight_layout()
    # single feature and target
    else:
        plt.scatter(data[[x_label]], data[[y_label]], linewidth = line_width, alpha = alpha, c = c, cmap = cmap)
        # plt.scatter(data[[x_label]], data[[y_label]], linewidth = line_width, alpha = alpha, c = c, cmap = cmap, s = data[s] / 100, label = s)
        # plt.set_xlim(min(time), max(time))
        # makes assumptions about the amounts being processed - x axis is in 10k+ and y axis single digits
        # but still a more adaptive way of plotting the labels
        plt.axis([
            int(round(data[[x_label]].min().values[0]*0.9, -3)),
            int(round(data[[x_label]].max().values[0]*1.04, -3)), 
            math.floor(data[[y_label]].min().values[0] - 1),  
            math.ceil(data[[y_label]].max().values[0]) + 1])
    
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(show_grid)
        plt.title(f'{y_label} according to {str.lower(x_label)}', y=1.2)
    
    return fig

def generate_projections(word_list, embeddings, show = False, 
                 c = None, s = None, height = 6, width = 8, line_width = 1, alpha = 0.4, 
                 style = 'ggplot', cmap = 'jet', subplots = False, show_grid = True, 
                 title_size = 14, label_size = 10):
    """
    Generate a plot based on input data.

    Parameters:
    - data: Input data.
    - x_label (optional): Label for the x-axis. Defaults to None.
    - y_label (optional): Label for the y-axis. Defaults to None.
    - show (bool, optional): Whether to display the plot. Defaults to False.
    - c (optional): Color parameter. Defaults to None.
    - s (optional): Size parameter. Defaults to None.
    - height (int, optional): Height of the plot. Defaults to 6.
    - width (int, optional): Width of the plot. Defaults to 8.
    - line_width (int, optional): Line width. Defaults to 1.
    - alpha (float, optional): Transparency of markers. Defaults to 0.4.
    - style (str, optional): Plot style. Defaults to 'ggplot'.
    - cmap (str, optional): Color map. Defaults to 'jet'.
    - subplots (bool, optional): Whether to use subplots. Defaults to False.
    - show_grid (bool, optional): Whether to display grid lines. Defaults to True.
    - title_size (int, optional): Font size of the title. Defaults to 14.
    - label_size (int, optional): Font size of labels. Defaults to 10.

    Returns:
    plt.Figure: Generated plot figure.
    """
    
    plt.rc('font', size=12)
    plt.rc('axes', labelsize = label_size, titlesize = title_size)
    plt.rc('legend', fontsize=12)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(1, 1, 1)

    # add a param to use different styles etc
    # color_map = 'viridis'
    # color = 'k'
    plt.style.use(style)
    x_min, x_max, y_min, y_max = 0, 0, 0, 0

    for i, label in enumerate(word_list):
        x, y = embeddings[i]

        if x < x_min:
            x_min = x
        elif x > x_max:
            x_max = x

        if y < y_min:
            y_min = y
        elif y > y_max:
            y_max = y

        # print (label, " : ", x, " " , y) # uncomment to see the detailed vector info
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2),
                    textcoords='offset points', ha='right', va='bottom')

    # makes assumptions about the amounts being processed - x axis is in 10k+ and y axis single digits
    # but still a more adaptive way of plotting the labels
    plt.axis([x_min-1, x_max+1, y_min-1, y_max+1])
    plt.grid(show_grid)
    plt.title('Word embeddings')
    
    return fig

def save_fig(fig_path, fig_name, fig, tight_layout = True, fig_extension = 'png', resolution = 300, show = False):
    """
    Save a generated plot figure.

    Parameters:
    - fig_path: Path to save the figure.
    - fig_name: Name of the figure file.
    - fig: Generated plot figure.
    - tight_layout (bool, optional): Whether to apply tight layout. Defaults to True.
    - fig_extension (str, optional): File extension of the figure. Defaults to 'png'.
    - resolution (int, optional): Resolution of the figure. Defaults to 300.
    - show (bool, optional): Whether to display the saved figure. Defaults to False.
    """
    if len(fig_name.split('.')) > 1:
        fig_extension = fig_name.split('.')[1]
        fig_name = fig_name.split('.')[0]
    
    path = f'{fig_path}{os.sep}{fig_name}'
    if tight_layout:
        plt.tight_layout()
    
    try:
        with open(f'{path}.pickle','wb') as file:
            pickle.dump(plt.gcf(),file)
        fig.savefig(f'{path}.{fig_extension}', format = fig_extension, dpi = resolution)
    except Exception as e:
        print(f'Saving the figure failed: \n\t{e}')

def add_predictions(fig, x_line, predicted_line, test_value, prediction, color = 'red', marker = 'kX', linewidth = 1, fontsize = 10, y = 0.8):
    """
    Add predictions to a plot figure.

    Parameters:
    - fig: Input plot figure.
    - x_line: X-axis values.
    - predicted_line: Predicted values.
    - test_value: Test values.
    - prediction: Predicted value.
    - color (str, optional): Color of the predictions. Defaults to 'red'.
    - marker (str, optional): Marker style. Defaults to 'kX'.
    - linewidth (int, optional): Line width. Defaults to 1.
    - fontsize (int, optional): Font size. Defaults to 10.
    - y (float, optional): Y-coordinate. Defaults to 0.8.

    Returns:
    plt.Figure: Updated plot figure.
    """

    plt.figure(fig.number)
    plt.plot(x_line, predicted_line, color = color, linewidth = linewidth)
    plt.plot(test_value, prediction, marker, label='prediction')
    plt.suptitle(f'Predicted {prediction[0][0]} from {test_value[0]}', fontsize = fontsize, y = y)

    return fig

def shuffle_and_split_data(data, test_ratio):
    """
    Shuffle and split input data into training and test sets.

    Parameters:
    - data: Input data.
    - test_ratio: Ratio of the test set.

    Returns:
    Tuple: Training and test sets.
    """

    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def is_id_in_test_set(identifier, test_ratio):
    """
    Determine if an identifier belongs to the test set based on its hash.

    Parameters:
    - identifier: Identifier value.
    - test_ratio: Ratio of the test set.

    Returns:
    bool: True if the identifier belongs to the test set, False otherwise.
    """
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    """
    Split data into training and test sets based on an identifier column.

    Parameters:
    - data: Input data.
    - test_ratio: Ratio of the test set.
    - id_column: Identifier column name.

    Returns:
    Tuple: Training and test sets.
    """
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

def stratified_split(data, label, splits = 10, size = 0.2, state = 42):

    return train_test_split(data, test_size = size, stratify = data[label], random_state = state)

def word_similarity(input_word, input_word2 = None, approach = 'wup'):
    """
    Measure the similarity between words using WordNet.

    Parameters:
    - input_word: Input word.
    - input_word2: Second input word (optional). Defaults to None.
    - approach: Similarity approach ('wup' or 'path'). Defaults to 'wup'.

    Returns:
    str: String indicating the similarity between words.
    """
    if input_word2:
        # comparing two provided words
        word1 = wn.synset(wn.synsets(input_word, pos=wn.NOUN)[0].name())
        word2 = wn.synset(wn.synsets(input_word2, pos=wn.NOUN)[0].name())
    else:
        # compare similarity with one word and it's synonym
        synsets = [x.name() for x in wn.synsets('language', pos=wn.NOUN)][:2]
        word1 = wn.synset(synsets[0])
        word2 = wn.synset(synsets[1])
    if approach == 'wup':
        return '{}<===>{}: {:.3f}'.format(word1.name(), word2.name(), wn.wup_similarity(word1, word2))
    elif approach == 'path':
        return '{}<===>{}: {:.3f}'.format(word1.name(), word2.name(), wn.path_similarity(word1, word2))

def tfidf(keys = [], docs = [], cleaned = False, lang = 'en', lang_name = 'english'):
    """
    Compute TF-IDF scores for given documents.

    Parameters:
    - keys: Keywords for fetching documents (optional).
    - docs: List of documents (optional).
    - cleaned (bool, optional): Whether the documents are pre-cleaned. Defaults to False.
    - lang: Language for Wikipedia search. Defaults to 'en'.
    - lang_name: Language name for NLTK stopwords. Defaults to 'english'.

    Returns:
    Dict: Dictionary containing TF-IDF scores.
    """
    DF, tf_idf, words, word_counts, cleaned = {}, {}, [], [], []
    words_count, doc_id, = [], 0
    nltk.download('punkt')
    nltk.download('stopwords')
    sww = sw.words()

    if not docs and not keys:
        # just retrieve some random ones
        keys = [ 'journal', 'diary', 'notebook']

    if not docs:
        # get some data using the key
        wikipedia.set_lang(lang)
        for key in keys:
            try:
                docs.append(wikipedia.page(key, auto_suggest = False).content)
            except wikipedia.exceptions.DisambiguationError as e:
                print(f'{key} is not available. Changing {key} to {e.options[0]}')
                docs.append(wikipedia.page(e.options[0]).content)
    doc_count = len(docs)
    for doc in docs:
        cleaned.append([t.lower() for t in word_tokenize(re.sub(r'[^a-zA-Z ]',' ',doc)) if not t.lower() in sww and t.lower()])
        for term in np.unique(cleaned[-1]):
            try:
                DF[term] +=1
            except:
                DF[term] =1

    for index in range(cleaned):
        item = cleaned[index]
        word_counts = len(item)
        counter = Counter(item)
        for word in np.unique(item):
            tf = counter[word]/word_counts

            # calculate Inverse Document Frequency
            idf = math.log(doc_count/(DF[word]+1)) +1
            # calculate TF-IDF
            if word in tf_idf:
                tf_idf[index, word] += float(tf*idf)
            else:
                tf_idf[index, word] = float(tf*idf)
    return tf_idf

def find_synonym(word, words, model = None):
    """
    Find synonyms for a given word.

    Parameters:
    - word: Input word.
    - words: List of words to search for synonyms.
    - model: Pre-trained word embedding model (optional).

    Returns:
    Tuple: The synonym and it's cosine similarity
    """
    if not model:
        model = gen_api.load("glove-wiki-gigaword-50")
    similarity = {}
    for term in words:
        if term not in similarity and term != word:
         similarity[term] = model.similarity(word, term)
    return sorted(similarity.items(), key = lambda val: val[1], reverse=True)[0]

def prepare_cbow_batch(data_temp, voc_size):
    """
    Prepare a one-hot encoded batch for cbow

    Parameters:
        data_temp 
        voc_size

    Returns:
    NP.Array:
    NP.Array: 
    """
    inputs = []
    labels = np.zeros((len(data_temp), voc_size), dtype=np.int64)

    for i in range(len(data_temp)):
        # ont-hot input - context
        input_temp = []
        for cw in data_temp[i][0]:
          cw_onehot=[0]*voc_size
          cw_onehot[cw]=1
          input_temp.append(cw_onehot)
        inputs.append(input_temp)

        # centre
        labels[i][data_temp[i][1] - 1] = 1

    return np.array(inputs), labels

def prepare_cbow_batch(data_temp, voc_size):
    """
    A preparatory function for 
    """
    inputs = []
    labels = []

    for i in range(len(data_temp)):
        # ont-hot input - context
        input_temp = []
        for cw in data_temp[i][0]:
          cw_onehot=[0]*voc_size
          cw_onehot[cw]=1
          input_temp.append(cw_onehot)
        inputs.append(input_temp)

        # centre
        labels.append(data_temp[i][1])

    return np.array(inputs), np.array(labels, dtype = np.int64)

def add_paddings(word, max_num=5):
    diff = max_num - len(word)
    return word+'P'*diff

def make_seq2seq_batch(seq_data, max_word_len = 5, dict_len:int = None, char_array = None):

    encoder_input_batch = []
    decoder_input_batch = []
    target_batch = []
    # Generate unique tokens list

    if not char_array:
        chars = []
        for seq in seq_data:
            chars += list(seq[0])
            chars += list(seq[1])
        # To simplify the question, we put all characters (including input and output) into one set
        char_array = list(set(chars))

        # special tokens are required
        # B: Beginning of Sequence
        # E: Ending of Sequence
        # P: Padding of Sequence - for different size input
        # U: Unknown element of Sequence - for different size input
        char_array.append('B')
        char_array.append('E')
        char_array.append('P')
        char_array.append('U')

    num_dict = prepare_vocab_dict(char_array)

    if not dict_len:
        dict_len = len(num_dict)

    for seq in seq_data:
        # Input for encoder cell, convert to vector
        input_word = add_paddings(seq[0], max_word_len)
        en_input_data = [num_dict[n] for n in input_word]
        # Input for decoder cell, Add 'B' at the beginning of the sequence data
        de_input_data  = [num_dict[n] for n in ('B'+ seq[1])]

        target = [num_dict[n] for n in (seq[1] + 'E')]

        # Convert each character vector to one-hot encoding data
        encoder_input_batch.append(np.eye(dict_len)[en_input_data])
        decoder_input_batch.append(np.eye(dict_len)[de_input_data])

        target_batch.append(target)
    return char_array, encoder_input_batch, decoder_input_batch, target_batch