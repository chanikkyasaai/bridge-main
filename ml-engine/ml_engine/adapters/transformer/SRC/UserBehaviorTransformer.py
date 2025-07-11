import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import torch
import torch.nn as nn

from InputEmbedding import InputEmbeddings
from PositionalEncoding import PositionalEncoding

class UserBehaviorTransformer(nn.Module):

  def __init__(self,action_vocab_size:int,embedding_size:int,num_heads:int,num_layers:int,ff_dim:int=128,dropout:float=0.1,max_seq_len:int=500):

    super().__init__()
    self.embedding=InputEmbeddings(action_count=action_vocab_size,
                                   embedding_size=embedding_size)
    self.positional_encoding=PositionalEncoding(
        embedding_size=embedding_size,
        dropout=dropout,
        sequence_max_length=max_seq_len)

    encoder_layer=nn.TransformerEncoderLayer(
        d_model=embedding_size, ##input features
        nhead=num_heads, ##number of heads / partitioning of input vector
        dim_feedforward=ff_dim, ##feed forward neural network dimensions
        dropout=dropout, ##The dropout rate normalization
        ##batch_first=True ##batch_first=True means input shape is (batch_size, seq_len, embedding_size) - allowing parallelization  
        ##Have to adapt positional encoding and input embeddings to work with this - not yet done that
    )

    self.transformer_encoder=nn.TransformerEncoder(
        encoder_layer=encoder_layer,
        num_layers=num_layers,
        enable_nested_tensor=True
    )

    self.classifier=nn.Sequential(
        nn.Linear(embedding_size,128),  ##projecting to a space of 128 size
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128,2) ##Legit or suspicious behavior
    )

  def forward(self,input_seq):
    x=self.embedding(input_seq)  ##initially input_seq will be of (batch_size,seq_len,embedding_size)
    ##The next line is not needed as the InputEmbeddings already returns the correct shape due to batch_first=True

    x=x.transpose(0,1)  ##convert input_seq t0 (seq_len,batch_size,embedding_size) for positional encoding

    x=self.positional_encoding(x)
    x=self.transformer_encoder(x)
    ##Global averaging across sequence of use to determine if that sequence was legitimate or not
    x=x.mean(dim=0)  ##input_sequence (batch_size,embedding_size)

    logits=self.classifier(x)
    return logits

