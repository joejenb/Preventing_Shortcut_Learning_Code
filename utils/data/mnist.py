import os
import lightly

import numpy as np
from PIL import Image

import torch
import torchvision
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils
from torch.utils.data import Dataset

from tqdm import tqdm


def color_grayscale_arr(arr, red=True):
  """Converts grayscale image to either red or green"""
  assert arr.ndim == 2
  dtype = arr.dtype
  h, w = arr.shape
  arr = np.reshape(arr, [h, w, 1])
  if red:
    arr = np.concatenate([arr,
                          np.zeros((h, w, 2), dtype=dtype)], axis=2)
  else:
    arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                          arr,
                          np.zeros((h, w, 1), dtype=dtype)], axis=2)
  return arr


class ColouredMNIST(datasets.VisionDataset):
  """
  Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf
  Args:
    root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
    env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
    transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.
  """
  def __init__(self, root='datasets/', env='train1', transform=None, target_transform=None):
    super(ColouredMNIST, self).__init__(root, transform=transform,
                                target_transform=target_transform)

    self.prepare_colored_mnist()
    if env in ['train1', 'train2', 'test']:
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColouredMNIST', env) + '.pt')
    elif env == 'all_train':
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColouredMNIST', 'train1.pt')) + \
                               torch.load(os.path.join(self.root, 'ColouredMNIST', 'train2.pt'))
    else:
      raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, test, and all_train')

  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data_label_tuples[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.data_label_tuples)

  def prepare_colored_mnist(self):
    colored_mnist_dir = os.path.join(self.root, 'ColouredMNIST')
    if os.path.exists(os.path.join(colored_mnist_dir, 'train1.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'train2.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'test.pt')):
      print('Colored MNIST dataset already exists')
      return

    print('Preparing Colored MNIST')
    train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

    train1_set = []
    train2_set = []
    test_set = []
    for idx, (im, label) in enumerate(train_mnist):
      if idx % 10000 == 0:
        print(f'Converting image {idx}/{len(train_mnist)}')
      im_array = np.array(im)

      # Assign a binary label y to the image based on the digit
      binary_label = 0 if label < 5 else 1

      # Flip label with 25% probability
      if np.random.uniform() < 0.25:
        binary_label = binary_label ^ 1

      # Color the image either red or green according to its possibly flipped label
      color_red = binary_label == 0

      # Flip the color with a probability e that depends on the environment
      if idx < 20000:
        # 20% in the first training environment
        if np.random.uniform() < 0.2:
          color_red = not color_red
      elif idx < 40000:
        # 10% in the first training environment
        if np.random.uniform() < 0.1:
          color_red = not color_red
      else:
        # 90% in the test environment
        if np.random.uniform() < 0.9:
          color_red = not color_red

      colored_arr = color_grayscale_arr(im_array, red=color_red)

      if idx < 20000:
        train1_set.append((Image.fromarray(colored_arr), binary_label))
      elif idx < 40000:
        train2_set.append((Image.fromarray(colored_arr), binary_label))
      else:
        test_set.append((Image.fromarray(colored_arr), binary_label))

      # Debug
      # print('original label', type(label), label)
      # print('binary label', binary_label)
      # print('assigned color', 'red' if color_red else 'green')
      # plt.imshow(colored_arr)
      # plt.show()
      # break

    #dataset_utils.makedir_exist_ok(colored_mnist_dir)
    torch.save(train1_set, os.path.join(colored_mnist_dir, 'train1.pt'))
    torch.save(train2_set, os.path.join(colored_mnist_dir, 'train2.pt'))
    torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))



class SubClassDataset(Dataset):
    def __init__(self, dataset, classes):
        self.dataset = dataset
        self.classes = classes
        print('Subsampling dataset...')
        self.indices = [i for i, (_, y) in enumerate(tqdm(dataset)) if y in classes]
        print('Done, {}/{} examples left'.format(len(self.indices), len(self.dataset)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        return x, self.classes.index(y)


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (*self.dataset[idx], idx)


class ColouredMNIST10(Dataset):
    def __init__(self, dataset, classes=None, colors=[0, 1], std=0, train=False):
        self.dataset = dataset
        self.colors = colors
        self.to_pil = torchvision.transforms.ToPILImage()
        self.train = train
        self.classes = 10
        self.normalise = torchvision.transforms.Normalize(
            mean=lightly.data.collate.imagenet_normalize['mean'],
            std=lightly.data.collate.imagenet_normalize['std'],
        )
        #if classes is None:
        #    classes = max([y for _, y in dataset]) + 1

        if isinstance(colors, torch.Tensor):
            self.colors = colors
        elif isinstance(colors, list):
            #For each class and each depth generate a value between 0 and 1
            #For each class have a specific colour -> random so could sometimes be the same?
            self.colors = torch.Tensor(classes, 3, 1, 1).uniform_(colors[0], colors[1])
        else:
            raise ValueError('Unsupported colors!')
        #For every item in dataset generate some random noise
        self.perturb = std * torch.randn(len(self.dataset), 3, 1, 1)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label, fname = self.dataset[idx]
        color_img = (self.colors[label] + self.perturb[idx]).clamp(0, 1) * img 
        #print(color_img.max())
        #print(color_img.max())
        if self.train:
          color_img = self.to_pil(color_img)
        else:
          color_img = self.normalise(color_img)
        #color_img = (self.colors[label] + self.perturb[idx]).clamp(0, 1) * img
        return color_img, label, fname