import torch.nn as nn
import io
from string import punctuation
from collections import Counter
import torch 
import numpy as np
#########################################################################
def featuresPadding(sentence,sequenceLength):
  features=np.zeros((len(sentence),sequenceLength),dtype=int)
  for i,row in enumerate(sentence):
    features[i,-len(row):]=np.array(row)[:sequenceLength]
  return features
############################################################################
def tokenize_sentence(sentence,vocab_to_int):
  sentence=sentence.lower()
  sentence=''.join([letter for letter in sentence if letter not in punctuation])
  test_words=sentence.split()
  tokens=[]
  sample=[]
  for word in test_words:
      if word in vocab_to_int:
          sample.append(word)
  tokens.append([vocab_to_int[word] for word in sample])
  return tokens
##############################################################################
def predict(model,sentence, vocab_to_int,sequence_length=200):
    test_ints=tokenize_sentence(sentence,vocab_to_int)
    features=featuresPadding(test_ints,sequence_length)
    feature_tensor=torch.from_numpy(features)
    batch_size=feature_tensor.size(0)
    
    hidden_state=model.initialize_hidden_state(batch_size)
    feature_tensor=feature_tensor.long()
    output,hidden_state=model(feature_tensor,hidden_state)
    prediction=torch.round(output.squeeze())
    if(prediction.item()==0):
       resultString="{:.4f}  Negative sentence!".format(output.item())
    else:
       resultString="{:.4f}  Positive sentence!".format(output.item())
    return resultString
########################################################################
def createData():
    vocab_to_int = np.load('vocab_to_int.npy').item()
    return vocab_to_int
#############################################################
class SentimentNetwork(nn.Module):
  def __init__(self,vocabulary_size,output_size,embedding_dimension,hidden_dimension,number_of_layers,dropout_probability=0.5):
    super(SentimentNetwork,self).__init__()
    self.output_size=output_size
    self.number_of_layers=number_of_layers
    self.hidden_dimension=hidden_dimension
    self.embedding=nn.Embedding(vocabulary_size,embedding_dimension)
    self.lstm=nn.LSTM(embedding_dimension,hidden_dimension,number_of_layers,dropout=dropout_probability,batch_first=True)
    self.dropout=nn.Dropout(dropout_probability)
    self.finalLayer=nn.Linear(hidden_dimension,output_size)
    self.sigmoid=nn.Sigmoid()
   
  def forward(self,x,hidden):
    batch_size=x.size(0)
    embedding_output=self.embedding(x)
    lstm_output,hidden=self.lstm(embedding_output,hidden)
    lstm_output=lstm_output.contiguous().view(-1,self.hidden_dimension)
    output=self.dropout(lstm_output)
    output=self.finalLayer(output)
    sigmoid_output=self.sigmoid(output)
    sigmoid_output=sigmoid_output.view(batch_size,-1)
    sigmoid_output=sigmoid_output[:,-1]
    return sigmoid_output,hidden
  
  def initialize_hidden_state(self,batch_size):
    weight=next(self.parameters()).data    
    hidden=(weight.new(self.number_of_layers,batch_size,self.hidden_dimension).zero_(),weight.new(self.number_of_layers,batch_size,self.hidden_dimension).zero_())
    return hidden
############################################################################################
from torch.optim import Optimizer
import  math
class AdamOptimizerAlgorithm(Optimizer):
  
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(AdamOptimizerAlgorithm, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization in case of initial state
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                #following is just a formula implementaion of the algorithm
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                b_1 = 1 - beta1 ** state['step']
                b_2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(b_2) / b_1
                
                p.data.addcdiv_(-step_size, exp_avg, denom)
#####################################################################################                
def buildNetwork(vocab_to_int):
  vocabulary_size=len(vocab_to_int)+1
  output_size=1
  embedding_dimension=200
  hidden_dimension=256
  number_of_layers=2
  model=SentimentNetwork(vocabulary_size,output_size,embedding_dimension,hidden_dimension,number_of_layers)
  learning_rate=0.003
  criterion=nn.MSELoss()
  optimizer=AdamOptimizerAlgorithm(model.parameters(),lr=learning_rate)
  return model
#########################################################################################
def load_checkpoint(filepath,vocab_to_int):
    model=buildNetwork(vocab_to_int)
    model.load_state_dict(torch.load(filepath,map_location='cpu'))
    model.cpu()
    return model
#############################################################################################
def getSentimentPredictionResult(inputReview):
    vocab_to_int=createData()
    model=load_checkpoint('SentimentAnalysisCheckpoint.pth',vocab_to_int)
    seq_length=200
    return predict(model,inputReview,vocab_to_int,seq_length)
