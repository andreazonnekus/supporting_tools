"""
Reference: https://docs.opencv.org/4.9.0/d8/d4b/tutorial_py_knn_opencv.html
Dataset: 10.24432/C5ZP40
"""
from PIL import Image
from dotenv import load_dotenv
import os, sys, pytesseract, cv2, pandas, statistics
from pytesseract import Output
import numpy as np

class KNN_SLATE:
    def __init__(self):
        load_dotenv()

        self.MODEL_PATH = os.environ.get('MODEL_PATH')
        if not os.path.exists(self.MODEL_PATH):
            print('The model path doesn\'t exist. Trying to create it...')
            try:
                os.makedirs(self.MODEL_PATH, exist_ok = True)
            except Exception as e:
                print(e)
                exit()


    def main(self):
        
        if len(sys.argv) > 1:
            if sys.argv[2] == 'generate':
                generate()
            elif sys.argv[2] == 'prep':
                prep()
            elif sys.argv[2] == 'identify':
                identify()

        # prep(knn_model)
        identify()

        return 0

    def generate(name = None):
        is_file = False
        # Load the data and convert the letters to numbers
        data= np.loadtxt(os.path.join('assets','train','letter-recognition.data'), dtype= 'float32', delimiter = ',',
                            converters= {0: lambda ch: ord(ch)-ord('A')})

        # Split the dataset in two, with 10000 samples each for training and test sets
        train, test = np.vsplit(data,2)

        # Split trainData and testData into features and responses
        responses, trainData = np.hsplit(train,[1])
        labels, testData = np.hsplit(test,[1])

        # Initiate the kNN, classify, measure accuracy
        knn = cv2.ml.KNearest_create()
        knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)

        ret, result, neighbours, dist = knn.findNearest(testData, k=5)
        correct = np.count_nonzero(result == labels)
        accuracy = correct*100.0/10000
        print( accuracy )
        print( labels )

        while name is None or is_file is False:
            name = input('\nProvide filename for the model:\n')

            if os.path.isfile(os.path.join(self.MODEL_PATH, name)):
                overwrite = input('\nType \'y\' to overwrite the existing file, or anything else to return to the previous prompt:\n')
                if overwrite == 'y':
                    is_file = True
                else:
                    name = None
            else:
                is_file = True

        knn.save(os.path.join(self.MODEL_PATH, name)) 
        if os.path.isfile(os.path.join(self.MODEL_PATH, name)):
            print(f'Model saved at {os.path.join(self.MODEL_PATH, name)}')

    def prep():
        knn = cv2.ml.KNearest.load()

    def identify():
        shift = lambda x: x-8

        features = pandas.DataFrame(
            {
                'x-box': [],
                'y-box': [],
                'width': [],
                'high':  [],
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

        while name is None or is_file is False:
            name = input('\nProvide a valid file to analyse:')
            
            if os.path.isfile(os.path.join('asset', 'input', f'{name}.jpg')):
                is_file = True

        filename = os.path.join('assets', 'input',f'{name}.jpg')
        img = cv2.imread(filename)
        img.resize(16, 16, 3)

        final_img = img.copy()
        mod_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        mod_img = mod_img.astype('uint8')

        thresh = cv2.threshold(mod_img, 146, 255, cv2.THRESH_BINARY)[1]

        letter = pandas.DataFrame(np.asarray(thresh))
        np.savetxt(os.path.join(self.MODEL_PATH, 'letter'), thresh, delimiter=',', fmt='%d')
        onpix = int(np.count_nonzero(thresh==0)/17) # everything was scaled to 15 max
        

        zeros = np.where(letter == 0)
        x_vals = shift(zeros[1])
        y_vals = shift(zeros[0])

        # x_max, y_max = zeros[1].max(), 16 - zeros[0].min()
        x_max, y_max = zeros[1].max(), zeros[0].max()
        # x, y = zeros[1].min(), 16 - zeros[0].max() - 1 # algorithm counts from the bottom
        x, y = zeros[1].min(), zeros[0].min() # algorithm counts from the bottom

        # inclusive height and width
        width = x_max - x + 1
        height = y_max - y

        cv2.rectangle(final_img, (x,y), (x_max, y_max), (255, 0, 255), 1)

        if x and y:
            feature_row.extend((x, y, width, height, onpix , int(x_vals.mean()), int(y_vals.mean()), int(np.square(x_vals).mean()), int(np.square(y_vals).mean())))
            print(feature_row)
        
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