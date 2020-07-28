# #########################################################################################################################################################
'''

Builds CellNET dataset from the hundred thousand covid19 images from https://www.rxrx.ai/rxrx19

The original dataset has five channels for each image (one for each treatment of the cells with a chemical). Here, we convert each channel into its own label,
and for any image from any channel, try to predict which channel it belongs to (what kind of chemical treatment the image is showing). We can then try
to use this huge dataset of single-channel images for 5-label classification tasks.

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

# open the dataframe of covid19 metadata (calculated earlier)
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
df = pandas.DataFrame({'experiment':experiments, 'plate':plates, 'well':wells, 'site':sites, 'channel':channels})
df['file_present'] = df[['experiment', 'plate', 'well', 'site', 'channel']].apply(lambda x: checkFile(*x), axis=1)
df = df[df['file_present'] == 1]
df.to_csv("valid_covid19_images.csv")

# comment out the section above and just use these cached images if you want to re-run this script
df = pandas.read_csv("valid_covid19_images.csv")
df = df.sample(n=2500).reset_index()
print(df.head())

os.system("mkdir covid_mini")
os.system("mkdir covid_mini/ch1")
os.system("mkdir covid_mini/ch2")
os.system("mkdir covid_mini/ch3")
os.system("mkdir covid_mini/ch4")
os.system("mkdir covid_mini/ch5")

def getFile(df, idx):
    path = '/mnt/fs03/shared/datasets/RxRx19/RxRx19a/images'
    entry = df.iloc[idx]
    file = '{}/{}/Plate{}/{}_s{}_w{}.png'.format(path, 
                                                  entry['experiment'], 
                                                  entry['plate'], 
                                                  entry['well'], 
                                                  entry['site'],
                                                  entry['channel'])
    print(file)
    channel = str(entry['channel'])
    os.system("cp " + file + " covid_mini/ch" + channel)

for i in list(range(len(df))):
    getFile(df, i)




