from PIL import Image
from dotenv import load_dotenv
import os, sys, pytesseract, cv2, pandas
from pytesseract import Output
import numpy as np

load_dotenv()

def main() -> int:
    try:
        pytesseract.pytesseract.tesseract_cmd = os.environ.get('TESSERACT_PATH')
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
    rect_d=[] #store the dimensions of my bounding boxes
    final = ''
    min_cont_area=5

    filename = os.path.join('assets', 'input','notepad_scanned.jpg')
    img = cv2.imread(filename)
    final_img = img.copy()
    mod_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)


    # knn_model = cv2.ml.KNearest.load(os.path.join(os.environ.get("MODEL_PATH"),'knn_writing'))

    # # mod_img = cv2.copyMakeBorder(img, 2, 2, 2, 2, borderType=cv2.BORDER_CONSTANT)
    # # cv2.imshow("Output: ", mod_img)
    # # cv2.imshow("Output: ", img)
    # cv2.waitKey(0)

    # mod_img = cv2.medianBlur(mod_img,3)
    mod_img = cv2.bilateralFilter(src=mod_img, d=9, sigmaColor=9, sigmaSpace=7)
    cv2.imshow("Output: ", mod_img)
    cv2.waitKey(0)

    mod_img = cv2.Canny(mod_img, 127, 255)

    print(mod_img)
    cv2.waitKey(0)

    mod_img = mod_img.astype('uint8')

    other, mod_img = cv2.threshold(mod_img,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)

    cv2.imshow("Thresh: ", mod_img)
    cv2.waitKey(0)

    contours, _ = cv2.findContours(mod_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    cont_img = mod_img.copy()

    cv2.drawContours(final_img, contours, -1, (0, 0, 255), cv2.FILLED, 8);

    # cv2.imshow(final_img)
    # cv2.imshow('Final Result',cont_img)
    cv2.imwrite('Final Result.jpg',final_img)

    cv2.waitKey(0)
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