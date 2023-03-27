"""
Contains functions for training/testing results vizualizations
"""

import matplotlib.pyplot as plt
from typing import Dict

def plot_results(results: Dict):
    '''
    Creates plots for training and testing results during training session

    args:   results(Dict): Dictionary that contains the training session data

    return: plots of the data n
    '''
    epochs = [i+1 for i in range(len(results['train_acc']))]

    plt.figure(figsize=(7,5))

    #plot loss
    plt.subplot(1,2,1)
    plt.title("Loss")
    plt.xlabel('Epochs')
    plt.plot(epochs,results['train_loss'])
    plt.plot(epochs,results['test_loss'])
    plt.legend(['Train loss','Test loss']);

    #plot accuracy
    plt.subplot(1,2,2)
    plt.title("Accuracy")
    plt.xlabel('Epochs')
    plt.plot(epochs,results['train_acc'])
    plt.plot(epochs,results['test_acc'])
    plt.legend(['Train accuracy','Test accuracy']);

    plt.show()