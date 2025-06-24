import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


from UserBehaviorTransformer import UserBehaviorTransformer

class UserBehaviorClassifier:

  def __init__(self,model,lr=1e-3,device='cpu'):
    self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    self.model=model
    self.model=self.model.to(device)
    self.optimizer=optim.Adam(model.parameters(),lr=lr)
    self.lossFunction=nn.CrossEntropyLoss()
    
  ##dataset will be of dimensions (num_samples,seq_len)
  def train(self,dataset:TensorDataset,epochs:int=100,batch_size:int=32):
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True)
    self.model.train()
    total_loss=0.0
    for epoch in range(epochs):
      epoch_loss=0.0
      for inputs,labels in dataloader:
        inputs=inputs.to(self.device)
        labels=labels.to(self.device)
        self.optimizer.zero_grad() ##clears gradient before backward pass
        logits=self.model(inputs)
        loss=self.lossFunction(logits,labels)
        loss.backward()
        self.optimizer.step()
        epoch_loss+=loss.item()
      avg_loss=epoch_loss/len(dataloader)
      total_loss+=epoch_loss
      print(f"Epoch {epoch+1}/{epochs} - Loss : {avg_loss:.4f}")

  def test(self,test_dataset:TensorDataset,batch_size:int=32):
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
    self.model.eval()

    correct=0
    total=0
    with torch.no_grad():
      for inputs,labels in test_loader:
        inputs=inputs.to(self.device)
        labels=labels.to(self.device)
        logits=self.model(inputs)
        probs=torch.softmax(logits,dim=1)
        preds=torch.argmax(probs,dim=1)
        total+=labels.size(0)
        correct+=(preds==labels).sum().item()
    accuracy=correct/total
    print(f"Test Accuracy: {accuracy:.4f}")
    returnList=[(idx,prob,pred) for idx,(prob,pred) in enumerate(zip(probs,preds))]
    return {"Accuracy":accuracy,"Probabilities":returnList}

  def predict(self,X):
    self.model.eval()
    with torch.no_grad():
      logits=self.model(X)
      probs=torch.softmax(logits,dim=1)
      preds=torch.argmax(probs,dim=1)
      return {"Probabilities":probs,"Predictions":preds}
  
        