import torch
import PIL
from PIL import Image
import torch.utils.data as data
from torch.utils.data import DistributedSampler
from torchvision import transforms
import utils.misc as utils

def My_loader(path):
    return PIL.Image.open(path).convert('RGB')

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, txt_dir, image_path, transform=None, target_transform=None, loader=My_loader):
        data_txt = open(txt_dir, 'r')
        imgs = []
        for line in data_txt:
            line = line.strip()
            words = line.split(' ')
            imgs.append((words[0], int(words[1].strip())))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = My_loader
        self.image_path = image_path

    def __len__(self):

        return len(self.imgs)

    def __getitem__(self, index):
        img_name, label = self.imgs[index]
        # label = list(map(int, label))
        # print label
        # print type(label)
        #img = self.loader('/home/vipl/llh/food101_finetuning/food101_vgg/origal_data/images/'+img_name.replace("\\","/"))
        img = self.loader(self.image_path + img_name)

        # print img
        if self.transform is not None:
            img = self.transform(img)
            # print img.size()
            # label =torch.Tensor(label)

            # print label.size()
        return img, label
        # if the label is the single-label it can be the int
        # if the multilabel can be the list to torch.tensor

def build_dataset(args):
    normalize = transforms.Normalize(mean=[0.5457954, 0.44430383, 0.34424934],
                                     std=[0.23273608, 0.24383051, 0.24237761])
    # transforms of train dataset
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # default value is 0.5
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.126, saturation=0.5),
        transforms.Resize((550, 550)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    # transforms of test dataset
    test_transforms = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    train_dir = "./data/Food2k_complete/train_finetune.txt"
    train_image_path = "./data/Food2k_complete/"
    test_dir = "./data/Food2k_complete/val_finetune.txt"
    test_image_path = "./data/Food2k_complete/"

    train_dataset = MyDataset(txt_dir=train_dir, image_path=train_image_path, transform=train_transforms)
    test_dataset = MyDataset(txt_dir=test_dir, image_path=test_image_path, transform=test_transforms)

    if args.distributed:
        sampler_train = DistributedSampler(train_dataset)
        sampler_val = DistributedSampler(test_dataset, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(test_dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn,
                                               num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, args.batch_size//2, sampler=sampler_val, collate_fn=utils.collate_fn,
                                              drop_last=False, num_workers=args.num_workers)
    return train_dataset, train_loader, test_dataset, test_loader, sampler_train, sampler_val