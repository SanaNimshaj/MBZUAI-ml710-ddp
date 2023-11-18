import json
import csv
import torch
import random

def LOG2CSV(datalist, csv_file, flag = 'a'):
    '''
    datalist: List of elements to be written
    '''
    with open(csv_file, flag) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(datalist)
    csvFile.close()

def LOG2TXT(text, file_path, flag = 'a', console= True):
    '''
    text: python object with stats to be logged
    '''
    text = str(text)
    with open(file_path, 'a', buffering=1) as txt_file:
        if console: print(text)
        print(text, file=txt_file)


def is_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False


def LOG2DICTXT(dic, file_path, flag = 'a', console= True):
    '''
    stats: dictionary object with stats to be logged
    '''
    with open(file_path, 'a', buffering=1) as txt_file:
        if console: print(json.dumps(dic))
        print(json.dumps(dic), file=txt_file)



def START_SEED(seed=6):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True