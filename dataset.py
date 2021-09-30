import torch
from torch.utils.data import Dataset
import scipy.io
from skimage import io, transform

class SynthText(Dataset):
    """ Load the 800k Pre-Generated SynthText Dataset.
        See README.txt for ground truth format:
        https://www.robots.ox.ac.uk/~vgg/data/scenetext/readme.txt

        load_ram allows you to load the entire dataset into memory
        if you think you have sufficient storage.
    """
    def __init__(self, root, transform=None, load_ram=False):
        super(SynthText, self).__init__()

        self.root = root + '/' if root[-1] != '/' else root
        self.gt = scipy.io.loadmat(self.root + 'gt.mat')
        self.transform = transform
        self.load_ram = load_ram

    def __len__(self):
        return len(self.gt['imnames'][0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = io.imread(self.root + self.gt['imnames'][0][idx][0])
        word_bb = self.gt['wordBB'][0][idx]
        char_bb = self.gt['charBB'][0][idx]
        txt = self.gt['txt'][0][idx]

        if self.transform:
            image = self.transform(image)

        return image, word_bb, char_bb, txt
