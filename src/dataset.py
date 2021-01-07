import os, glob
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root,target, form):
        self.root = root
        if type(target) == str : 
            self.data_list = [x for x in glob.glob(os.path.join(root, target, form), recursive=False) if not os.path.isdir(x)]
        elif type(target) == list : 
            self.data_list = []
            for i in target : 
                self.data_list = self.data_list + [x for x in glob.glob(os.path.join(root, target, form), recursive=False) if not os.path.isdir(x)]
        else : 
            raise Exception('Unsupported type for target')

    def __getitem__(self, index):
        data_item = self.data_list[index]

        data = data_item

        return data

    def __len__(self):
        return len(self.data_list)


