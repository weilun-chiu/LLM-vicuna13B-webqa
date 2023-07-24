import json
import configargparse
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from io import BytesIO
import base64

def readImage(iid, fp, lineidx):
    """Read image"""
    fp.seek(lineidx[int(iid)%10000000])
    readline = fp.readline()
    imgid, img_base64 = readline.strip().split('\t')
    im = Image.open(BytesIO(base64.b64decode(img_base64)))
    return im

def create_loader(
    data: dict,
    params: configargparse.Namespace,
    is_train: bool,
    min_batch_size: int = 1,
    shortest_first: bool = False,
    is_aug: bool = False,
):
    """Creates dataset and loader."""
    # data = json.loads(data)
    data = sorted(
        data.items()
    )

    dataset = WebQADataset(data, is_train)

    loader = DataLoader(
        dataset=dataset,
        num_workers=1
    )

    return dataset, loader 


class WebQADataset(Dataset):
    def __init__(self, data: dict, is_train: bool):
        """
        :param dict data- The json data file dictionary
        :param Namespace params- Training options
        """
        self.data = data
        self.is_train = is_train

    def __len__(self):
        """Returns the number of examples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx: str):
        """Retrieves an item from the dataset given the index"""
        
        Q = self.data[idx][1]["Q"]
        A = self.data[idx][1]["A"]
        split = self.data[idx][1]["split"]
        Qcate = self.data[idx][1]["Qcate"]
        Guid = self.data[idx][1]["Guid"]
        
        # topic = self.data[idx][1]["topic"]
        
        img_posFacts = self.data[idx][1]["img_posFacts"]
        img_negFacts = self.data[idx][1]["img_negFacts"]
        txt_posFacts = self.data[idx][1]["txt_posFacts"]
        txt_negFacts = self.data[idx][1]["txt_negFacts"]
        
        img_posFacts_data = [( \
            imgData['image_id'], \
            imgData['title'], \
            imgData['caption']) \
            for imgData in img_posFacts]
        img_negFacts_data = [( \
            imgData['image_id'], \
            imgData['title'], \
            imgData['caption']) \
            for imgData in img_negFacts]
        
        txt_posFacts_data = [(txtData['title'], txtData['fact']) for txtData in txt_posFacts]
        txt_negFacts_data = [(txtData['title'], txtData['fact']) for txtData in txt_negFacts]
        return Q, A, split, Qcate, Guid, img_posFacts_data, img_negFacts_data, txt_posFacts_data, txt_negFacts_data
