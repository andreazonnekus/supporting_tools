try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

def ocr_core(filename):
    """
    This function will handle the core OCR processing of images.
    """
    text = pytesseract.image_to_string(Image.open(filename),lang='eng+jpn+chi_trad+afr+kor')  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    return text

def make_txt(string, name):
    txt_output = open('txt\\' + name.split('.')[0] + '.txt','w')
    txt_output.write(string)
    txt_output.close()
    return 'txt\\' + name.split('.')[0] + '.txt'
    
print('image:')
img = input()
print('img at ' + make_txt(ocr_core('img/'+ img),img))

