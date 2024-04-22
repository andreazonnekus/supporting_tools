
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from classes.utils import make_seq2seq_batch, add_paddings
from sklearn.metrics import accuracy_score

class Bidirectional_RNN(nn.Module):
    def __init__(self, n_hidden, n_class, n_input):
        super(Bidirectional_RNN, self).__init__()

        # RNN encoder.
        self.rnn_encoder = nn.RNN(n_input, n_hidden, batch_first= True, bidirectional= True)

        # Dropout can be applied as a layer on the output of the RNN (see below in forward()).
        # Previously we applied it as a parameter of the nn.RNN layer ('dropout=0.2')
        self.dropout_encoder = nn.Dropout(0.1)

        # RNN decoder - 2* to account for directions
        self.rnn_decoder = nn.RNN(n_input, n_hidden*2, batch_first= True)
        self.dropout_decoder = nn.Dropout(0.1)
        self.linear = nn.Linear(n_hidden*2,n_class)

    def forward(self, x_encoder, x_decoder):

        # "hidden" containing the hidden state for t = seq_len.
        _,hidden = self.rnn_encoder(x_encoder)


        # print(f'hidden {hidden.shape}')
        # Applying dropout layer on the output of the RNN
        hidden = self.dropout_encoder(hidden)

        bi_output = torch.cat((hidden[0], hidden[1]))

        bi_output = bi_output.view(1,hidden.shape[1],hidden.shape[-1]*2)

        # [IMPORTANT] Setting "hidden" as inital_state of rnn_decoder
        decoder_output,_ = self.rnn_decoder(x_decoder,bi_output)
        # print(f'decoded {decoder_output.shape}')
        # Applying dropout layer on output of RNN
        decoder_output = self.dropout_decoder(decoder_output)
        # print(f'decoded dropout {decoder_output.shape}')

        # prediction_output_before_softmax = self.linear(decoder_output)
        # output_after_softmax = torch.log_softmax(prediction_output_before_softmax,dim=-1)
        # Since nn.CrossEntropyLoss combines LogSoftmax and NLLLoss for us, we only need the prediction_output_before_softmax
        output = self.linear(decoder_output)

        return output
    
    def predict(self, word, char_array):
        # add padding
        word = add_paddings(word)

        # Setting each character of predicted as 'U' (Unknown)
        # ['king'(padded), 'UU']
        seq_data = [word, 'U' * 2]

        _, encoder_input_batch, decoder_input_batch, _ = make_seq2seq_batch([seq_data], char_array= char_array)
        encoder_input_torch = torch.from_numpy(np.array(encoder_input_batch)).float().to(device)
        decoder_input_torch = torch.from_numpy(np.array(decoder_input_batch)).float().to(device)

        model.eval()
        outputs = model(encoder_input_torch, decoder_input_torch)
        predicted = torch.argmax(outputs, -1)
        first_token = char_arr[predicted.cpu().numpy()[0][0]]

        # predict second token
        seq_data[1] = first_token + 'U'

        _, encoder_input_batch, decoder_input_batch, _ = make_seq2seq_batch([seq_data], char_array= char_array)
        encoder_input_torch = torch.from_numpy(np.array(encoder_input_batch)).float().to(device)
        decoder_input_torch = torch.from_numpy(np.array(decoder_input_batch)).float().to(device)

        model.eval()
        outputs = model(encoder_input_torch, decoder_input_torch)
        predicted = torch.argmax(outputs, -1)
        second_token = char_arr[predicted.cpu().numpy()[0][1]]

        return first_token + second_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Setting Hyperparameters
learning_rate = 0.01
n_hidden = 128
total_epoch = 200

# Sequence data
seq_data = [['ace', '01'], ['jack', '11'],
            ['queen', '12'], ['king', '13']]

max_input_words_amount = 5
max_output_words_amount = 3

char_arr, encoder_input_batch, decoder_input_batch, target_batch, = make_seq2seq_batch(seq_data, max_input_words_amount)
encoder_input_torch = torch.from_numpy(np.array(encoder_input_batch)).float().to(device)
decoder_input_torch = torch.from_numpy(np.array(decoder_input_batch)).float().to(device)
target_batch_torch = torch.from_numpy(np.array(target_batch)).view(-1).to(device).long()

n_class = len(char_arr)
n_input = n_class

model = Bidirectional_RNN(n_hidden, n_class, n_input).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(total_epoch):  # loop over the dataset multiple times

    model.train()
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(encoder_input_torch, decoder_input_torch)
    
    loss = criterion(outputs.view(-1,outputs.size(-1)), target_batch_torch)
    loss.backward()
    optimizer.step()

    if epoch%10==9:
        print('Epoch: %d, loss: %.5f' %(epoch + 1, loss.item()))

print('=== Prediction result ===')
print('ace ->', model.predict('ace', char_array= char_arr))
print('jack ->', model.predict('jack', char_array= char_arr))
print('queen ->', model.predict('queen', char_array= char_arr))
print('king ->', model.predict('king', char_array= char_arr))