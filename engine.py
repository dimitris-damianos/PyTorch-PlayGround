"""
Contains functions for training and testing a model
"""

import torch
from tqdm.auto import tqdm
from typing import Dict,List,Tuple

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer = torch.optim.Optimizer):
    '''
    Implements the training for 1 epoch

    args:   model: a PyTorch model to be trained
            dataloader: Dataloader for the model to be trained on
            loss_fn: loss function to minimize
            optimizer: optimize function to help minimize the cost

    return: train loss and train accuracy 
    '''

    model.train() #set model on train mode
    train_loss, train_acc = 0,0

    #start training
    for batch, (X,y) in enumerate(dataloader):
        y_pred = model(X)
        loss = loss_fn(y_pred,y)
        
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(y_pred,dim=1)

        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    
    #adjust metrics
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc

def test_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module):
    '''
    Implements the training for 1 epoch

    args:   model: a PyTorch model to be trained
            dataloader: Dataloader for the model to be trained on
            loss_fn: loss function to minimize
            optimizer: optimize function to help minimize the cost

    return: train loss and train accuracy 
    '''

    model.eval() #set model on train mode
    test_loss, test_acc = 0,0

    #start training
    for batch, (X,y) in enumerate(dataloader):
        y_pred = model(X)
        loss = loss_fn(y_pred,y)
        '''
        loss function must be CrossEntropyLoss() (in:vector)
        '''
        
        test_loss += loss.item()

        y_pred_class = torch.argmax(y_pred,dim=1)

        test_acc += (y_pred_class == y).sum().item()/len(y_pred)
    
    #adjust metrics
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc

def train(model:torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int)->Dict[str,List]:
    """
    Trains and tests a model

    args:   model
            train_dataloader
            test_dataloader
            optimizer
            loss functions
            epochs

    return: Dictionary of training and testing loss and accuracy
    """

    results = {
        "train_loss" :[],
        "train_acc":[],
        "test_loss":[],
        "test_acc":[]
    }

    #train model for the selected number of epochs
    print("Train session starting")
    for epoch in tqdm(range(epochs)):
        train_loss,train_acc = train_step(model=model,
                                        dataloader=train_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer)
        
        test_loss,test_acc = test_step(model=model,
                                        dataloader=train_dataloader,
                                        loss_fn=loss_fn)
        
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
        )
        
        #update dictionary
        results["test_acc"].append(test_acc)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["train_loss"].append(train_loss)
    
    return results
        
        





