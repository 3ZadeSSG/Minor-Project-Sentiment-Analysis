import torch.nn as nn
import io
from string import punctuation
from collections import Counter
import torch 
import numpy as np

def pad_features(reviews_ints, seq_length):
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
    return features

def tokenize_review(test_review,vocab_to_int):
    test_review = test_review.lower()
    test_text = ''.join([c for c in test_review if c not in punctuation])
    test_words = test_text.split()
    test_ints = []
    sample=[]
    for word in test_words:
        if word in vocab_to_int:
            sample.append(word)
    test_ints.append([vocab_to_int[word] for word in sample])
    return test_ints


def predict(net, test_review, vocab_to_int,sequence_length=200):
    net.eval()
    test_ints = tokenize_review(test_review,vocab_to_int)
    seq_length=sequence_length
    features = pad_features(test_ints, seq_length)
    feature_tensor = torch.from_numpy(features)
    batch_size = feature_tensor.size(0)
    h = net.init_hidden(batch_size)
    output, h = net(feature_tensor.long(), h)
    pred = torch.round(output.squeeze()) 
    resultString='Prediction value: {:.6f}\t'.format(output.item())
    if(pred.item()==1):
        resultString=resultString+"Positive statement detected!"
    else:
        resultString=resultString+"Negative statement detected."

    return resultString
########################################################################
def createData():
    with open('reviews.txt', 'r') as f:
        reviews = f.read()
    with open('labels.txt', 'r') as f:
        labels = f.read()
    reviews = reviews.lower() # lowercase, standardize
    all_text = ''.join([c for c in reviews if c not in punctuation])
    reviews_split = all_text.split('\n')
    all_text = ' '.join(reviews_split)
    words = all_text.split()
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    return vocab_to_int

#############################################################
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(SentimentRNN, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        

    def forward(self, x, hidden):
        
        batch_size = x.size(0)
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sig(out)
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden
############################################################################################
def buildNetwork(vocab_to_int):
    vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding + our word tokens
    output_size = 1
    embedding_dim = 200
    hidden_dim = 256
    n_layers = 2
    net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    lr=0.003
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    return net
  

def load_checkpoint(filepath,vocab_to_int):
    model=buildNetwork(vocab_to_int)
    model.load_state_dict(torch.load(filepath))
    return model
	
#############################################################################################
def getSentimentPredictionResult(inputReview):
    vocab_to_int=createData()
    model=load_checkpoint('CHECKPOINT99.pth',vocab_to_int)
    seq_length=200
    return predict(model,inputReview,vocab_to_int,seq_length)



#predict(net, test_review_pos, seq_length)












'''
def get_model():
	checkpoint_path='PyTorchCheckpoint'
	checkpoint=torch.load(checkpoint_path,map_location='cpu')
  	model=checkpoint["model"]
  	model.load_state_dict(checkpoint['state_dict'])
  	return model

 #def getFeatures():
'''