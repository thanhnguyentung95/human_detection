from torch.utils.data import Dataset
import os


class Data(Dataset):
    def __init__(self, path, img_size=640):
        # Assert if file exist
        assert os.path.isfile(path), '"{}" does not exist'.format(path)

        # Get path dataset folder
        dirname = os.path.dirname(path)

        # Load image paths
        with open(path) as f:
            self.img_paths = f.readlines()
            self.img_paths = [p.replace('./', dirname).strip() for p in self.img_paths]

        self.len = len(img_paths)
        # self.label_path
    
    
    def __len__(self):
        return self.len


    def __getitem__(self, item):
        pass

