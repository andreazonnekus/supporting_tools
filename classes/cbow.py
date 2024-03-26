import torch.nn as nn
import random, pickle, os, torch
import numpy as np
from joblib import dump, load
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from random import shuffle

from utils import *

np.random.seed(42)

class CBOW(nn.Module):
    def __init__(self, classes_dim, window_dim, embedding_dim, num_iterations = 1000, learning_rate = 0.01):
        super(CBOW, self).__init__()

        self.learning_rate = learning_rate
        self.iterations = num_iterations

        self.linear1 = nn.Linear(classes_dim, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim * window_size, classes_dim)
        self.activation = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()
        # self.loss = nn.NLLLoss()
    
    def forward(self, x):
        l1 = self.linear1(torch.sum(x, dim=1))

        # get the output of l1
        out = self.linear2(l1)
        return torch.log_softmax(out, dim=1)

cbow = []

sentences = ["he likes cat",
             "he likes dog",
             "he likes animal",
             "dog cat animal",
             "she likes cat",
             "she dislikes dog",
             "cat likes fish",
             "cat likes milk",
             "dog likes bone",
             "dog dislikes fish",
             "dog likes milk",
             "she likes movie",
             "she likes music",
             "he likes game",
             "he likes movie",
             "cat dislikes dog"]

# convert all sentences to unique word list
word_list = " ".join(sentences).split()
word_list = list(set(word_list))

# make dictionary so that we can reference each index of unique word
word_dict = {w: i for i, w in enumerate(word_list)}

for sentence in sentences:
    sentence = sentence.split()
    for i in range(len(sentence)):
        centre = word_dict[sentence[i]]
        if i > 0 and i < len(sentence)-1:
            context = [word_dict[sentence[i - 1]], word_dict[sentence[i + 1]]]
        elif i == 0:
            context = [word_dict[sentence[i + 1]], word_dict[sentence[i + 1]]]
        else:
            context = [word_dict[sentence[i - 1]], word_dict[sentence[i - 1]]]

        cbow.append([context, centre])

# learning rate
voc_size = len(word_list)
learning_rate = 0.01
batch_size = 4
embedding_size = len(cbow[0])
window_size = 1
no_of_epochs = 5000

# initialise a cbow model
model = CBOW(voc_size, window_size, embedding_size, no_of_epochs, learning_rate)
optimiser = optim.SGD(model.parameters(), lr = learning_rate)
no_of_epochs = 5000

for epoch in range(no_of_epochs):

    # shuffle the training set to make each epoch's batch different, you can also skip this step
    shuffle(cbow)
    loss_sum = 0

    for ind in range(0, len(cbow),batch_size):
        data_temp = cbow[ind : min(ind+batch_size, len(cbow))]
        inputs_temp, labels_temp = prepare_batch(data_temp, voc_size)
        
        inputs_torch = torch.from_numpy(inputs_temp).float()
        labels_torch = torch.from_numpy(labels_temp).long()
        
        model.train() # mode = True by default

        # set the gradients to zero
        optimiser.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs_torch)
        loss = model.loss(outputs, labels_torch)
        loss.backward()

        optimiser.step() # back propagation

        loss_sum += loss.item()

    if epoch % 500 == 499:
        print(labels_torch.shape, outputs.size())
        print('Epoch: %d, loss: %.4f' %(epoch + 1, loss_sum))

figure = generate_projections(word_list, model.embedding.weight.data)
fig_path = os.path.join('assets', 'output')

save_fig(fig_path, 'cbow', figure)