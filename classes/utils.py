import os, sys, cv2, math
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import matplotlib.pyplot as plt

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

def generate_fig(data, x_label = None, y_label = None, show = False, height = 10, width = 5, line_width = 1, alpha = 1, style = 'ggplot', subplots = False, show_grid = True, title_size = 14, label_size = 10):
    plt.rc('font', size=12)
    plt.rc('axes', labelsize = label_size, titlesize = title_size)
    plt.rc('legend', fontsize=12)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    fig = plt.figure(figsize=(height, width))
    ax = fig.add_subplot(1, 1, 1)

    # add a param to use different styles etc
    # color_map = 'viridis'
    # color = 'k'
    plt.style.use(style)
    plt.scatter(data[[x_label]], data[[y_label]], linewidth = line_width, alpha = alpha)

    if not x_label:
        x_label = data.columns[1]
    if not y_label:
        y_label = data.columns[2]
    
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
    plt.title(f'{y_label} according to {str.lower(x_label)}')
    
    return fig

def save_fig(fig_path, fig_name, fig, tight_layout = True, fig_extension = 'png', resolution = 300, show = False):
    
    # if show:
    #     plt.show()

    if len(fig_name.split('.')) > 1:
        fig_name = fig_name.split('.')[0]
        fig_extension = fig_name.split('.')[1]
    
    path = f'{fig_path}{os.sep}{fig_name}.{fig_extension}'
    if tight_layout:
        plt.tight_layout()
    
    try:
        fig.savefig(path, format = fig_extension, dpi = resolution)
    except Exception as e:
        print(f'Saving the figure failed: \n\t{e}')

if __name__ == '__main__':
    sys.exit(main())