"""
Functions for downloading,transforming data, and creating dataloaders
"""

import os
import zipfile 
from pathlib import Path
import requests
import pandas as pd
import torch

from torchvision import datasets,transforms
from torch.utils.data import DataLoader, Dataset

###NOTE: multithreading locally causes error, is not currently implemented
NUM_WORKERS = os.cpu_count() #number of cpu available

def download_data(source:str,
                  destination:str,
                  remove_source:bool = True)->Path:
    '''
    Downloads zip file from source and unzips them into destination C:\torch_lib\directory

    args:   source (str): zip file link
            destination (str): target destination
            remove_source (bool): remove zip file after unzipping

    return: pathlib.Path to download path
    '''
    data_path = Path(r'C:\torch_lib')

    #check if directory exists
    if data_path.is_dir():
        print(f"[INFO] {data_path} directory already exists.")
    else:
        print(f"[INFO] {data_path} doesn't exist, creating one...")
        data_path.mkdir(parents=True,exist_ok=True)
    
    #download data
    target_file = Path(source).name
    with open(data_path/target_file,'wb') as f:
        request = requests.get(source)
        print(f"[INFO] Downloading {target_file} from {source}...")
        f.write(request.content)

    #unzip data
    with zipfile.ZipFile(data_path / target_file, 'r') as zip:
        print(f"[INFO] Unzipping {target_file}...")
        zip.extractall(data_path / destination)

    #remove zip file
    if remove_source:
        os.remove(data_path / target_file)

    return data_path / destination

def dir_dataloader(train_dir: str,
                      test_dir: str,
                      train_transform: transforms.Compose,
                      test_transform: transforms.Compose,
                      batch_size: int):
    '''
    Creates dataloaders for train and test data located locally

    args:   train_dir (str): train directory 
            test_dir (str): test directory
            transform: type of transform applied to data (transform pipeline)
    
    return: train dataloader, test dataloader, class names

    NOTE: the required form of data is NOT .csv, but in this directory form:
        train: 
            label1: img1
                    img2
                    img3
                    ...
            label2: img1
                    img2
                    ...
    '''

    #create datasets using ImageFolder and apply the transform
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir,transform=test_transform)

    #get class names
    class_names = train_data.classes

    #turn datasets into dataloaders
    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True)
    
    test_dataloader = DataLoader(test_data,
                                  batch_size=batch_size,
                                  shuffle=True)
    
    return train_dataloader, test_dataloader, class_names

def default_transformer(height: int,
                        width: int):
    '''
    Transformer that resizes images into height x width and turns them into Tensors
    '''
    return transforms.Compose([
        transforms.Resize((height,width)),
        transforms.ToTensor()
    ])

def augm_transformer(height:int,
                     width:int):
    '''
    Transformer that resizes images into height x width, turns thme into Tensors 
    an augments data by flipping them with possibilty p
    '''
    return transforms.Compose([
        transforms.Resize((height,width)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])


class custom_dataset(Dataset):
    '''
    Creates Datasets from pandas dataframes.
    '''
    def __init__(self,file_name):
        '''
        args:   file_name (str): file path
        
        NOTE: LABELS MUST BE ON COLUMN NAMED 'label'.
        '''
        file_df = pd.read_csv(file_name)
        rows,cols = file_df.shape

        labels = file_df['label'].values #list of values
        data = file_df.drop(['label'],axis=1).values #list of columns

        self.x_train = torch.tensor(data,dtype=torch.float32)
        self.y_train = torch.tensor(labels,dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]


def csv_dataloader(file_path:str,
                   batch_size:int=32):
    '''
    Creates dataloader from custom csv file.
    '''
    dataset = custom_dataset(file_name=file_path)
    class_names = []
    for label in dataset.y_train:
        if label not in class_names:
            class_names.append(label)
    print(f"Labels type is:{type(class_names[0].item())}")
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=True),class_names






    