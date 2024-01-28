"""
Reference: https://docs.opencv.org/4.9.0/d8/d4b/tutorial_py_knn_opencv.html
Dataset: 10.24432/C5ZP40
"""

import numpy as np
import os, cv2

from dotenv import load_dotenv

load_dotenv()

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

knn.save(os.path.join(os.environ.get("MODEL_PATH"),'knn_writing')) 