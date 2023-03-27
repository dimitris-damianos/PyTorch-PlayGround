import data_setup
import models
import engine
import utils

import os
import torch
import matplotlib.pyplot as plt
import pandas as pd

#hyperparameters
EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

'''
#set up directories
train_dir = 'C:/torch_lib/pizza_steak_sushi/train'
test_dir = 'C:/torch_lib/pizza_steak_sushi/test'

train_transformer = data_setup.augm_transformer(64,64)
test_transformer = data_setup.default_transformer(64,64)

train_dataloader,test_dataloader,class_names = data_setup.dir_dataloader(
    train_dir=train_dir,
    test_dir=test_dir,
    train_transform=train_transformer,
    test_transform= test_transformer,
    batch_size=BATCH_SIZE
)
'''

train_dataloader, class_names = data_setup.csv_dataloader(file_path="DATA\mnist_train.csv",
                                                          batch_size=BATCH_SIZE)

test_dataloader, class_names = data_setup.csv_dataloader(file_path="DATA\mnist_test.csv",
                                                          batch_size=BATCH_SIZE)
class_names = [int(i) for i in class_names]

print(len(class_names))

model = models.DenseNN_h1(
    input_shape=784,
    hidden_shape=HIDDEN_UNITS,
    output_shape=len(class_names)
)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=LEARNING_RATE)

results = engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             optimizer=optimizer,
             loss_fn=loss_fn,
             epochs=EPOCHS)

utils.plot_results(results=results)
