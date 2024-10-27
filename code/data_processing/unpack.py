import pickle
import numpy as np
import pandas as pd
import os

FOLDER = 'cifar-10-batches-py/' # redefine if you store dataset in another folder
SAVE_PATH = 'data/'

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def create_save_dir():
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

def meta_to_csv(folder: str):
    batches_dict = unpickle(folder + 'batches.meta')
    to_csv_dict = {}

    for i in range(10):
        to_csv_dict[i] = batches_dict[b'label_names'][i]

    str_labels = [str(word) for word in to_csv_dict.values()]

    df = pd.DataFrame(data={
        'label': str_labels
    })
    create_save_dir()
    df.to_csv('data/labels.csv')

def train_to_csv(folder: str = FOLDER):
    ids = []
    labels = []
    uid_name_map = {}
    uid = 0

    for i in range(5):
        batch_dict = unpickle(folder + f'data_batch_{i + 1}')

        for j in range(len(batch_dict[b'labels'])):
            uid += 1
            uid_name_map[uid] = batch_dict[b'filenames'][j]
            labels.append(batch_dict[b'labels'][j])
            ids.append(uid)
    
    
    df = pd.DataFrame(data={
        'id': ids,
        'label': labels
    })

    create_save_dir()
    df.to_csv('data/train.csv')

    return uid_name_map


def test_to_csv(folder: str = FOLDER):
    test_dict = unpickle(folder + 'test_batch')
    ids = []
    labels = []
    uid_name_map = {}
    uid = 0

    for j in range(len(test_dict[b'labels'])):
        uid += 1
        uid_name_map[uid] = test_dict[b'filenames'][j]
        labels.append(test_dict[b'labels'][j])
        ids.append(uid)
    
    df = pd.DataFrame(data={
        'id': ids,
        'label': labels
    })

    create_save_dir()
    df.to_csv('data/test.csv')

    return uid_name_map

# train_id_map = train_to_csv()
# test_id_map = test_to_csv()

