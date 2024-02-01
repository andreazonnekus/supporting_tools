from PIL import Image
from dotenv import load_dotenv
import os, sys, pytesseract, cv2, pandas
import numpy as np

load_dotenv()


def main() -> int:

    # method = sys.argv[1], 
    # filename = sys.argv[2], 
    filename = os.path.join('assets', 'input','tesseract.jpg')
    img = cv2.imread(filename)

    box(img)

    return 0

def box(img):

    final_img = img.copy()
    mod_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    mod_img = mod_img.astype('uint8')

    thresh = cv2.threshold(mod_img, 146, 255, cv2.THRESH_BINARY)[1]

    
    letter = pandas.DataFrame(np.asarray(thresh))
    np.savetxt(os.path.join(MODEL_PATH, 'letter'), thresh, delimiter=',', fmt='%d')

    zeros = np.where(letter == 0)

    x_max, y_max = zeros[1].max(), zeros[0].max()
    x, y = zeros[1].min(), zeros[0].min() 

    cv2.rectangle(final_img, (x,y), (x_max, y_max), (255, 0, 255), 1);
    
    cv2.imwrite('Final Result.jpg',final_img)

if __name__ == '__main__':
    sys.exit(main())