
import torch.nn as nn
from torch import cat, from_numpy, argmax, device, cuda
from numpy import array
import torch.nn.functional as F
import torch.optim as optim
from utils import make_seq2seq_batch, add_paddings
from sklearn.metrics import accuracy_score

class Seq2Seq_Model(nn.Module):
    def __init__(self, n_hidden, n_class, n_input):
        super(Seq2Seq_Model, self).__init__()

        # RNN encoder.
        self.rnn_encoder = nn.RNN(n_input, n_hidden, batch_first= True, bidirectional= True)
        self.dropout_encoder = nn.Dropout(0.1)

        # RNN decoder
        self.rnn_decoder = nn.RNN(n_input, n_hidden, batch_first= True)
        self.dropout_decoder = nn.Dropout(0.1)
        self.linear = nn.Linear(n_hidden,n_class)

    def forward(self, x_encoder, x_decoder):

        _,hidden = self.rnn_encoder(x_encoder)
        hidden = self.dropout_encoder(hidden)

        dec_output = cat((hidden[0], hidden[1]))

        # [IMPORTANT] Setting "hidden" as inital_state of rnn_decoder
        decoder_output, _ = self.rnn_decoder(x_decoder,hidden)
        # print(f'decoded {decoder_output.shape}')
        # Applying dropout layer on output of RNN
        decoder_output = self.dropout_decoder(decoder_output)
        # print(f'decoded dropout {decoder_output.shape}')

        # prediction_output_before_softmax = self.linear(decoder_output)
        # output_after_softmax = torch.log_softmax(prediction_output_before_softmax,dim=-1)
        # Since nn.CrossEntropyLoss combines LogSoftmax and NLLLoss for us, we only need the prediction_output_before_softmax

        return self.linear(decoder_output)
    
    def predict(self, word, char_array):
        # add padding
        word = add_paddings(word)

        # Setting each character of predicted as 'U' (Unknown)
        # ['king'(padded), 'UU']
        seq_data = [word, 'U' * 2]

        _, _, encoder_input_batch, decoder_input_batch, = make_seq2seq_batch([seq_data], char_array= char_array)
        encoder_input_torch = from_numpy(array(encoder_input_batch)).float().to(device)
        decoder_input_torch = from_numpy(array(decoder_input_batch)).float().to(device)

        model.eval()
        outputs = model(encoder_input_torch, decoder_input_torch)
        predicted = argmax(outputs, -1)
        first_token = char_arr[predicted.cpu().numpy()[0][0]]

        # predict second token
        seq_data[1] = first_token + 'U'

        _, _, encoder_input_batch, decoder_input_batch, = make_seq2seq_batch([seq_data], char_array= char_array)
        encoder_input_torch = from_numpy(array(encoder_input_batch)).float().to(device)
        decoder_input_torch = from_numpy(array(decoder_input_batch)).float().to(device)

        model.eval()
        outputs = model(encoder_input_torch, decoder_input_torch)
        predicted = argmax(outputs, -1)
        second_token = char_arr[predicted.cpu().numpy()[0][1]]

        return first_token + second_token

device = device("cuda" if cuda.is_available() else "cpu")

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
encoder_input_torch = from_numpy(array(encoder_input_batch)).float().to(device)
decoder_input_torch = from_numpy(array(decoder_input_batch)).float().to(device)
target_batch_torch = from_numpy(array(target_batch)).view(-1).to(device).long()

n_class = len(char_arr)
n_input = n_class

model = Seq2Seq_Model(n_hidden, n_class, n_input).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(array(target_batch).shape)

for epoch in range(total_epoch):  # loop over the dataset multiple times

    model.train()
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(encoder_input_torch, decoder_input_torch)

    # squeeze, unsqueeze, shape to target batch and amount of classes
    outputs = F.interpolate(outputs.unsqueeze(0).unsqueeze(0), size=(len(target_batch) * len(target_batch[0]), n_class), mode='nearest').squeeze()
    
    loss = criterion(outputs, target_batch_torch)
    loss.backward()
    optimizer.step()

    if epoch%10==9:
        print('Epoch: %d, loss: %.5f' %(epoch + 1, loss.item()))

print('=== Prediction result ===')
print('ace ->', model.predict('ace', char_array= char_arr))
print('jack ->', model.predict('jack', char_array= char_arr))
print('queen ->', model.predict('queen', char_array= char_arr))
print('king ->', model.predict('king', char_array= char_arr))