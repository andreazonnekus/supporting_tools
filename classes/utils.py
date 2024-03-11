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

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

load_dotenv()

def main() -> int:
    # method = sys.argv[1], 
    # filename = sys.argv[2], 
    
    if len(sys.argv) > 1:
        img = None
        if len(sys.argv) > 2:
            img = sys.argv[2] if os.path.isfile(os.path.join('assets', 'input', sys.argv[2])) else None
            if not model:
                print('This model doesn\'t exist')

        if len(sys.argv) > 3:
            show = sys.argv[3] if os.path.isfile(os.path.join(model_path, sys.argv[3])) else True

        if len(sys.argv) > 4:
            input_dicts = sys.argv[4] # TODO: check dict structure if os.path.isfile(os.path.join(model_path, sys.argv[4])) else True

        if sys.argv[1] == 'box':
            box(img)
        elif sys.argv[1] == 'boxes':
            boxes(img)
        elif sys.argv[1] == 'prep':
            prep_input(img)
        elif sys.argv[1] == 'label':
            label_img(input_dicts, img, show)
    

    return 0

def box(img, show = False):
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

    # np.savetxt(os.path.join(os.path.realpath('.'), 'output', 'letter'), thresh, delimiter=',', fmt='%d')

    cv2.rectangle(img, (x,y), (x_max, y_max), (255, 0, 0), 1)

    if show:
        cv2.imshow('tst', img)
        cv2.waitKey(0)
        
        cv2.imwrite('Final Result.jpg',img)

    return (x, y), (x_max, y_max)

def boxes(img, show = False):
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
    for input_dict in input_dicts:
        if show:
                print(input_dict)
        cv2.rectangle(img, input_dict['mn'], input_dict['mx'], (255, 0, 0), 1)
        cv2.putText(img, str(input_dict['label']), (int(input_dict['mn'][0] +(input_dict['mx'][0] - input_dict['mn'][0]) / 4), input_dict['mx'][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        if show:
            cv2.imshow(name, img)
            cv2.waitKey(0)
        cv2.imwrite(os.path.join('assets', 'output', 'final.jpg'),img)

def generate_fig(data, x_label = None, y_label = None, show = False, 
                 c = None, s = None, height = 6, width = 8, line_width = 1, alpha = 0.4, 
                 style = 'ggplot', cmap = 'jet', subplots = False, show_grid = True, 
                 title_size = 14, label_size = 10):
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

    if not x_label and len(data.columns) > 2:
        x_label = data.columns[1]
    if not y_label and len(data.columns) > 2:
        y_label = data.columns[2]

    if not x_label and not y_label:
        data.hist()
    else:
        plt.scatter(data[[x_label]], data[[y_label]], linewidth = line_width, alpha = alpha, c = c, cmap = cmap, s = data[s] / 100, label = s)

    
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

def save_fig(fig_path, fig_name, fig, tight_layout = True, fig_extension = 'png', resolution = 300, show = False):
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

if __name__ == '__main__':
    sys.exit(main())

def add_predictions(fig, x_line, predicted_line, test_value, prediction, color = 'red', marker = 'kX', linewidth = 1, fontsize = 10, y = 0.8):
        x_min, x_max = fig.axes[0].get_xlim()

        plt.figure(fig.number)
        plt.plot(x_line, predicted_line, color = color, linewidth = linewidth)
        plt.plot(test_value, prediction, marker, label='prediction')
        plt.suptitle(f'Predicted {prediction[0][0]} from {test_value[0]}', fontsize = fontsize, y = y)

        return fig

def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

def stratified_split(data, label, splits = 10, size = 0.2, state = 42):
    return train_test_split(data, test_size = size, stratify = data[label], random_state = state)

def word_similarity(input_word, input_word2 = None, approach = 'wup'):
    if input_word2:
        # comparing two provided words
        word1 = wn.synset(wn.synsets(input_word1, pos=wn.NOUN)[0].name())
        word2 = wn.synset(wn.synsets(Input_word2, pos=wn.NOUN)[0].name())
    else:
        # compare similarity with one word and it's synonym
        synsets = [x.name() for x in wn.synsets('language', pos=wn.NOUN)][:2]
        word1 = wn.synset(synsets[0])
        word2 = wn.synset(synsets[1])
    if approach == 'wup':
        return '{}<===>{}: {:.3f}'.format(word1.name(), word2.name(), wn.wup_similarity(word1, word2))
    elif approach == 'path':
        return '{}<===>{}: {:.3f}'.format(word1.name(), word2.name(), wn.path_similarity(word1, word2))

def clean_and_tokenize_data(docs, lang_name = 'english'):
    tokenized_docs, tokenized_words = [], []
    
    for doc in docs:
        tokenized_docs.append([x.lower() for x in word_tokenize(doc) if x not in string.punctuation and x not in sw.words(lang_name)])
    
    return docs

def tfidf(keys = [], docs = [], cleaned = False, lang = 'en', lang_name = 'english'):
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
                docs.append(wikipedia.page(key).content)
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

    for item in cleaned:
        word_counts = len(item)
        counter = Counter(item)
        for word in np.unique(item):
        tf = counter[word]/word_counts

        # calculate Inverse Document Frequency
        idf = math.log(doc_count/(DF[word]+1)) +1
        # calculate TF-IDF
        if word in tf_idf:
            tf_idf[word] += float(tf*idf)
        else:
            tf_idf[word] = float(tf*idf)
    return tf_idf