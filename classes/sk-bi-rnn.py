import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pickle, os
import matplotlib.pyplot as plt
import pandas as pd
import utils

# You can enable GPU here (cuda); or just CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Bi_RNN_Model(nn.Module):
    def __init__(self, n_hidden, n_class, n_input):
        super(Bi_RNN_Model, self).__init__()

        # RNN encoder.
        self.rnn = nn.RNN(n_input, n_hidden, batch_first= True, bidirectional= True)
        self.linear = nn.Linear(n_hidden*2,n_class)

    def forward(self, x):

        # "hidden" containing the hidden state for t = seq_len.
        _,hidden = self.rnn(x)

        bi_output = torch.cat((hidden[0,:,:], hidden[1,:,:]), dim=1)

        return self.linear(bi_output)

categories = ['alt.atheism', 'comp.graphics', 'misc.forsale', 'sci.med', 'soc.religion.christian']

input_embeddings = pickle.load(open(os.path.join("assets", "train","embedded_docs.pkl"),"rb"))
label = pickle.load(open(os.path.join("assets", "train","labels.pkl"),"rb"))

train_embeddings, test_embeddings, train_label, test_label = train_test_split(input_embeddings,label,test_size = 0.2, random_state=0)

### Setting Hyperparameters

n_hidden = 128
total_epoch = 200
batch_size = 4

model_accuracies = np.zeros((1, total_epoch))
model_f1s = np.zeros((1, total_epoch))
# This should be the length of each sequence
seq_length = train_embeddings.shape[2]

# Sequence data

max_input_words_amount = 5
max_output_words_amount = 3

n_class = len(categories)
n_input = seq_length

input_test_torch = torch.from_numpy(test_embeddings).float().to(device)
target_test_torch = torch.from_numpy(test_label).view(-1).to(device)

criterion = nn.CrossEntropyLoss()
model = Bi_RNN_Model(n_hidden, n_class, n_input).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(total_epoch):
    train_loss = 0
    acc = 0
    f1 = 0
    for ind in range(0,train_embeddings.shape[0],batch_size):
        input_batch = train_embeddings[ind:min(ind+batch_size, train_embeddings.shape[0])]
        target_batch = train_label[ind:min(ind+batch_size, train_embeddings.shape[0])]
        input_batch_torch = torch.from_numpy(input_batch).float().to(device)
        target_batch_torch = torch.from_numpy(target_batch).view(-1).to(device)

        model.train()
        optimizer.zero_grad()
        outputs = model(input_batch_torch)
        loss = criterion(outputs, target_batch_torch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    outputs = model(input_test_torch)
    predicted = torch.argmax(outputs, 1)
    model_accuracies[1, epoch] = accuracy_score(target_test_torch.cpu().numpy(),predicted.cpu().numpy())
    model_f1s[1, epoch] = f1_score(target_test_torch, predicted, average='weighted')
    print('Epoch: %d, train loss: %.5f, f1: %.5f, acc: %.5f'%(epoch + 1, train_loss, model_f1s[1, epoch], model_accuracies[1, epoch]))

print('Finished Training')

## Prediction
model.eval()
outputs = model(input_batch_torch).squeeze(0)
predicted = torch.argmax(outputs, 1)

print(classification_report([categories[i] for i in test_label], [categories[i] for i in predicted.cpu().numpy()], digits=4))