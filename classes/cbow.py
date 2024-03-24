import torch.nn as nn
import random, pickle, os, torch
import numpy as np
from joblib import dump, load
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from random import shuffle

from utils import *

#We are defining the class of our model
class CBOW(nn.Module):
    def __init__(self, classes_dim, window_dim, embedding_dim, num_iterations = 1000, learning_rate = 0.01):
        super(CBOW, self).__init__()
        self.learning_rate = learning_rate
        self.iterations = num_iterations

        self.embedding = nn.Embedding(classes_dim,  embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 2*window_dim)
        self.linear2 = nn.Linear(2*window_dim, classes_dim)
        self.activation =  nn.ReLU()
        self.loss = nn.NLLLoss()
    
    def forward(self, x):
        embedding = self.embedding(x)

        # find the mean
        mean_embeddings = torch.mean(embedding, dim=1)
        l1 = self.linear1(mean_embeddings)

        # get the output of l1
        l1_out = self.activation(l1)
        out = self.linear2(self.activation(l1_out))

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
learning_rate = 0.1
batch_size = 4
embedding_size = len(cbow[0])
window_size = 1
no_of_epochs = 5000

model = CBOW(voc_size, window_size, embedding_size, no_of_epochs, learning_rate)
optimiser = optim.SGD(model.parameters(), lr = model.learning_rate)

for epoch in range(model.iterations):
    # shuffle the training set to make each epoch's batch different, you can also skip this step
    shuffle(cbow)
    loss_sum = 0

    for ind in range(0, len(cbow), batch_size):
        data_temp = cbow[ind : min(ind+batch_size, len(cbow))]

        inputs_temp, labels_temp = prepare_cbow_batch(data_temp, voc_size)
        inputs_torch = torch.from_numpy(inputs_temp).long()
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
        model.eval()
        pred_outputs = model(inputs_torch)
        predicted = torch.argmax(pred_outputs, 2)
        out = predicted.shape[1] - torch.argmax(torch.flip(torch.argmax(pred_outputs, 2), dims=[1]), 1)
        train_acc = accuracy_score(out.numpy(),(labels_torch.argmax(1)+1).numpy())
        print('%d, loss: %.3f, train_acc: %.3f' %(epoch+1, loss.item(), train_acc))

figure = generate_projections(word_list, model.embedding.weight.data)
fig_path = os.path.join('assets', 'output')

save_fig(fig_path, 'cbow', figure)