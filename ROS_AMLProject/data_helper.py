
import torch
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from dataset import Dataset, TestDataset, _dataset_info

NUM_WORKERS = 2 # 4 Was the default, but Colab kept suggested using 2

def get_train_dataloader(args,txt_file):


    img_transformer = get_train_transformers(args)
    name_train, labels_train = _dataset_info(txt_file)
    train_dataset = Dataset(name_train, labels_train, args.path_dataset, img_transformer=img_transformer)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    return loader


def get_val_dataloader(args,txt_file):

    names, labels = _dataset_info(txt_file)
    img_tr = get_test_transformer(args)
    test_dataset = TestDataset(names, labels,args.path_dataset, img_transformer=img_tr)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

    return loader


def get_train_transformers(args):

    img_tr = [transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (args.min_scale, args.max_scale))]

    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))
    if args.random_grayscale:
        img_tr.append(transforms.RandomGrayscale(args.random_grayscale))

    img_tr = img_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr)


def get_test_transformer(args):

    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_img(dataloader,class_names,batch_size=5):
  # Get a batch of training data
  inputs, classes = next(iter(dataloader))
  inputs, classes = inputs[0:batch_size], classes[0:batch_size]

  # Make a grid from batch
  out = torchvision.utils.make_grid(inputs)

  imshow(out, title=[class_names[x] for x in classes])

