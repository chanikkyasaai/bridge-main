import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


import math
import torch.nn as nn
import numpy as np



class InputEmbeddings(nn.Module):
  ##Action_count= no.of user actions to learn, embedding_size-dimension of vector to which it will be embedded
  def __init__(self,action_count:int,embedding_size:int):
    super().__init__()
    self.input_action_embeddings=nn.Embedding(num_embeddings=action_count,embedding_dim=embedding_size)
    self.embedding_size=embedding_size
  def forward(self,input_vector):
    return self.input_action_embeddings(input_vector)*math.sqrt(self.embedding_size)
    ##The multiplication by embedding_size/d_model - suggested by attention is all you need paper

