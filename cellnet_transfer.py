# #########################################################################################################################################################
'''

Builds CellNET models from the hundred thousand covid19 images from https://www.rxrx.ai/rxrx19

'''
# #########################################################################################################################################################

# PyTorch
from torchvision import datasets, models
import torch
from torch.utils.data import DataLoader, sampler, random_split, ConcatDataset, Dataset

# Data science tools
import numpy 
import pandas
import os
import random
import sys
import statistics
from sklearn.metrics import balanced_accuracy_score, accuracy_score

# local libraries
from models import *
from dataset_prep import *
from augmentations import *

# #########################################################################################################################################################
# BASIC CONFIGURATION
# #########################################################################################################################################################

batch_size = 8
epochs = 10
DATASET_SIZE = 50000
learning_rate = 1e-4
scoring = "_" + str(epochs) + "epochs_lr" + str(learning_rate).replace("0.", "_") 
device = "cuda"

calcStats = False       # turn this on if you want to calc mean and stddev of training dataset for normalization in transform

# used with calcStats to calculate the mean and stddev of the greyscale images for later normalization; you may need to update this for your dataset
stats_transforms = [
                    transforms.Grayscale(),
                    transforms.Resize([380,380]),
                    transforms.ToTensor()]

# choose a GPU from the command line arguments
gpu_num = int(sys.argv[1])
if cuda.is_available():
    torch.cuda.set_device(gpu_num)
    print("starting, using GPU " + str(gpu_num) + "...")
else:
    device = "cpu"
    print("starting, using CPU")

# #########################################################################################################################################################
# DATASET PREP
# #########################################################################################################################################################

# a custom class for the covid19 dataset
class RxRxDataset(Dataset):
    def __init__(self, df=None, data_path = '/mnt/fs03/shared/datasets/RxRx19/RxRx19a/images', train=False, size=380):
        assert df is not None, 'No df'
        self.df = df
        self.data_path = data_path

        if train:
            self.transform = transforms.Compose([
                #RGB(),
                transforms.RandomRotation(degrees=90),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                Grayscale(),
                transforms.Resize(size=[size,size]),
                transforms.ToTensor(),
                #transforms.Normalize([2.3584, 2.3584, 2.3584], [0.0913, 0.0913, 0.0913])
                transforms.Normalize([-25.1580], [0.5144])
            ])
        else:
            self.transform = transforms.Compose([
                #RGB(),
                Grayscale(),
                transforms.Resize(size=[size,size]),
                transforms.ToTensor(),
                #transforms.Normalize([2.3584, 2.3584, 2.3584], [0.0913, 0.0913, 0.0913])
                transforms.Normalize([-25.1580], [0.5144])
            ])        
            
            
    def __getitem__(self, idx):
        entry = self.df.iloc[idx]
        file = '{}/{}/Plate{}/{}_s{}_w{}.png'.format(self.data_path, 
                                                  entry['experiment'], 
                                                  entry['plate'], 
                                                  entry['well'], 
                                                  entry['site'],
                                                  entry['channel'])
        img = Image.open(file)
        img_tensor = self.transform(img)
        return img_tensor, (entry['channel'] - 1), file

    def __len__(self):
        return len(self.df)

# see if the file in the dataframe actually exists in the directory
def checkFile(experiment, plate, well, site, channel):
    data_path = '/mnt/fs03/shared/datasets/RxRx19/RxRx19a/images'
    file = '{}/{}/Plate{}/{}_s{}_w{}.png'.format(data_path, 
                                                experiment,
                                                plate,
                                                well, 
                                                site,
                                                channel)
    if os.path.exists(file):
        return 1
    else:
        print("missing ", file)
        return 0

# open the dataframe of covid19 metadata
df = pandas.read_csv("./metadata.csv")
#df = df.sample(frac=0.5)

# convert the dataframe into a dataframe with metadata where each image of five channels becomes five images of a single channel;
# this will be what we predict (which channel the image comes from). Hopefully useful for transfer learning from B&W datasets
experiment = list(df['experiment'])
plate = list(df['plate'])
well = list(df['well'])
site = list(df['site'])
channels = []
experiments = []
plates = []
wells = []
sites = []
for i in range(len(df)):
    channels.extend([1, 2, 3, 4, 5])
    experiments.extend([experiment[i]]*5)
    plates.extend([plate[i]]*5)
    wells.extend([well[i]]*5)
    sites.extend([site[i]]*5)

# create a dataframe of images that exist, after checking if they are in the dataset (not all the metadata was valid)
#df = pandas.DataFrame({'experiment':experiments, 'plate':plates, 'well':wells, 'site':sites, 'channel':channels})
#df['file_present'] = df[['experiment', 'plate', 'well', 'site', 'channel']].apply(lambda x: checkFile(*x), axis=1)
#df = df[df['file_present'] == 1]
#df.to_csv("valid_covid19_images.csv")
df = pandas.read_csv("valid_covid19_images.csv")
df = df.sample(n = DATASET_SIZE)
print("length of DataFrame: ", len(df))

# set the test-train group for each image (TODO: this is not really random)
def group(x):
    if x <= 3:
        return "holdout"
    elif x >= 25:
        return "eval"
    else:
        return "train"
df['group'] = df['plate'].apply(lambda x: group(x))

df_eval = df[df['group'] == 'eval']
df_holdout = df[df['group'] == 'holdout']
df_train = df[df['group'] == 'train']

# create the datasets for training, test, and holdout
print("creating training dataset...")
train_dataset = RxRxDataset(df=df_train)
print("creating eval dataset...")
eval_dataset = RxRxDataset(df=df_eval, train=False)
print("creating holdout dataset...")
holdout_dataset = RxRxDataset(df=df_holdout, train=False)

# find the mean and stddev for the training data, and quit, so these can be manually copied into the config file
if calcStats:
    print('calculating stats')
    loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    getMeanStddev(loader)

# Dataloader iterators; each should use certain plates only
dataloaders = {}
dataloaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
dataloaders['eval'] = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
dataloaders['holdout'] = DataLoader(holdout_dataset, batch_size=batch_size, shuffle=False)

# train and test the model
print("training model...")
#neurons = ResNet18ModelAllLayers(n_classes=4, learning_rate=learning_rate)
neurons = Vgg19OneChannelModelAllLayers(n_classes=5, learning_rate=learning_rate, pretrained=False)
#neurons = Vgg19ThreeChannelModelAllLayers(n_classes=5, learning_rate=learning_rate, pretrained=True)
#neurons.model.load_state_dict(torch.load("./cellnet_vgg19_rotations_covid_noImageNet.torch"))
model_options = {'name':'cellnet_vgg19_1channel_rotations_covid_withoutImageNet_' + str(DATASET_SIZE) + 'samples'}
model_options['file_label'] = scoring
neurons.train(dataloaders['train'], dataloaders['eval'], epochs, device, model_options)
neurons.test(dataloaders['eval'], device, model_options, None, "test")

# test the model on the global holdout
print("testing model on holdout...")
all_preds, all_targets, confidences, paths = neurons.test(dataloaders['holdout'], device, model_options, None, "holdout")
print("Weighted accuracy holdout: " + str(weighted_accuracy(all_preds, all_targets)))





