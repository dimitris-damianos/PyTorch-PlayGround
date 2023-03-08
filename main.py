import data_setup
import models
import engine

import os
import torch

#hyperparameters
EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.1

#set up directories
train_dir = 'C:/torch_lib/pizza_steak_sushi/train'
test_dir = 'C:/torch_lib/pizza_steak_sushi/test'

transformer = data_setup.default_transformer(64,64)

train_dataloader,test_dataloader,class_names = data_setup.create_dataloader(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=transformer,
    batch_size=BATCH_SIZE
)

model = models.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=LEARNING_RATE)

engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             optimizer=optimizer,
             loss_fn=loss_fn,
             epochs=EPOCHS)