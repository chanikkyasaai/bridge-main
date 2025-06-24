import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
  ##Dropout to randomly zero some elements
  def __init__(self,embedding_size:int,dropout:float=0.1,sequence_max_length:int=500):
    super().__init__()
    self.dropout=nn.Dropout(p=dropout)
    positional_encoding=torch.zeros(sequence_max_length,embedding_size)
    position=torch.arange(0,sequence_max_length,dtype=torch.float).unsqueeze(1)
    div_term=torch.exp(torch.arange(0,embedding_size,2).float()*(-math.log(10000.0)/embedding_size))
    positional_encoding[:,0::2]=torch.sin(position*div_term)
    positional_encoding[:,1::2]=torch.cos(position*div_term)
    positional_encoding=positional_encoding.unsqueeze(0).transpose(0,1)
    self.register_buffer('positional_encoding',positional_encoding)
  
  def forward(self,x):
    ##x:(seq_len,batch_size,embedding_size)
    x=x+self.positional_encoding[:x.size(0),:]
    return self.dropout(x)