from PIL import Image
from dotenv import load_dotenv
import os, sys, pytesseract, cv2, pandas, statistics
from pytesseract import Output
import numpy as np

load_dotenv()

MODEL_PATH = os.environ.get('MODEL_PATH')
TESSERACT_PATH = os.environ.get('TESSERACT_PATH')

def main() -> int:
    try:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        if (not os.path.exists(pytesseract.pytesseract.tesseract_cmd)) or (not os.path.isfile(pytesseract.pytesseract.tesseract_cmd)):
            raise Exception("Please supply a valid path for the Tesseract executable")
    except KeyError:
        print("Supply a path for the Tesseract executable")
        exit()
    except Exception as e:
        print(e)
        exit()
    
    # prep(knn_model)

    identify()

    return 0

def prep(model):
    knn = cv2.ml.KNearest.load()

def identify():
    shift = lambda x: x-8

    features = pandas.DataFrame(
        {
            'x-box': [],
            'y-box': [],
            'width': [],
            'high': [],
            'onpix': [],
            'x-bar': [],
            'y-bar': [],
            'x2bar': [],
            'y2bar': [],
            'xybar': [],
            'x2ybr': [],
            'xy2br': [],
            'x-ege': [],
            'xegvy': [],
            'y-ege': [],
            'yegvy': [],
        }
    )
    feature_row, x_pos, y_pos = [], [], []
    final = ''
    min_cont_area=5

    filename = os.path.join('assets', 'input','K.jpg')
    img = cv2.imread(filename)
    final_img = img.copy()
    mod_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    mod_img = mod_img.astype('uint8')

    thresh = cv2.threshold(mod_img, 146, 255, cv2.THRESH_BINARY)[1]

    # contours, _ = cv2.findContours(thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    contours = cv2.findContours(thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)[-2]

    cv2.drawContours(final_img, contours, -1, (255, 255, 255), cv2.FILLED);
    
    letter = pandas.DataFrame(np.asarray(thresh))
    np.savetxt(os.path.join(MODEL_PATH, 'letter'), thresh, delimiter=',', fmt='%d')
    onpix = int(np.count_nonzero(thresh==0)/17) # everything was scaled to 15 max

    x, x_bind, y_bind, last = letter.shape[1]-1, 0, 0, 0 # The first and last 
    y = letter.shape[0]-1
    
    # inclusive height and width
    height = y_bind-y
    width = x_bind-x + 2 

    zeros = np.where(letter == 0)
    x_vals = shift(zeros[1])
    y_vals = shift(zeros[0])

    x_max, y_max = zeros[1].max(), 16 - zeros[0].min()
    x, y = zeros[1].min(), 16 - zeros[0].max() - 1 # algorithm counts from the bottom

    width = x_max - x + 1
    height = y_max - y


    if x and y:
        feature_row.extend((x, y, width, height, onpix , int(x_vals.mean()), int(y_vals.mean()), int(np.square(x_vals).mean()), int(np.square(y_vals).mean())))
        print(feature_row)
    
    # Width of box

    # Heigth of box

    # Amount of pixels in box
    #feature_row.append(total_pixels)
    # cv2.imshow('Final Result',cont_img)
    cv2.imwrite('Final Result.jpg',final_img)

    cv2.destroyAllWindows()

    # ret,result,neighbours,dist = knn_model.findNearest(test,k=5)

    # print(result)

    # results = pytesseract.image_to_data(mod_img, output_type=Output.DATAFRAME).replace(' ', np.nan, inplace=False).dropna()
        
    # results.drop(results[results.conf < 10].index, inplace=True)

    # for i in range(0, len(results.text)):
    #     x = results.left.iloc[i]
    #     y = results.top.iloc[i]

    #     w = results.width.iloc[i]
    #     h = results.height.iloc[i]

    #     text = results.text.iloc[i]
    #     conf = int(results.conf.iloc[i])
        
    #     if conf > 60:
    #         text = "".join([c if c.isalnum() else ' ' for c in text]).strip()
    #         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #         cv2.putText(img, text, (x, y - 10), 
    # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

    # cv2.imshow("Output: ", img)
    # cv2.waitKey(0)

    # results.replace(' ', np.nan, inplace=False).dropna()
    # output = results.groupby(['page_num', 'par_num', 'block_num', 'line_num'], as_index=False)['text'].agg(lambda x: ' '.join(x)).reset_index()

    # for index, row in output.iterrows():
    #     final+=f'\n{row["text"].strip()}'

    # print(final)

if __name__ == '__main__':
    sys.exit(main())