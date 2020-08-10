# #####################################################################################################################
'''

This module is used to test one or more models+data specified in config files on a CV framework.

For the BioNic project, we will be running the following experiments across several biomedical datasets:
1. Investigate how transfer learning improves models (or not)
2. Investigate what are the best data pre-processing and augmentations to use
3. Investigate if transfer learning from CellNet is better than transfer learning from ImageNet
4. Use VennData to make measurements about the quality of our datasets, and see if we can get better results with that 
   knowledge

Future work for this project includes:
1. Implement something to automatically resize an image to a standard cell size 
2. Implement something to count whole cells on an image
3. Freeze and train top layer for 5 epochs, then try to retrain the whole thing for 45 epochs?
4. Active learning?

'''
# #####################################################################################################################

# PyTorch
from torchvision import datasets, models
import torch
from torch.utils.data import DataLoader, sampler, random_split, ConcatDataset

# image processing
from PIL import Image

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

# model config files
from augment_config import *
from cellnet_config import *
from internal_brightfield_config import *

# #####################################################################################################################
# BASIC CONFIGURATION
# #####################################################################################################################
batch_size = 8
epochs = 20             # default epochs (can be overriden in *_config.py for your models)
learning_rate = 1e-3
device = "cuda"

calcStats = False  # turn this on if you want to calc mean and stddev of training dataset for normalization in transform
cvFolds = 5        # how many cross validation folds to use
repeats = 4        # how many times you want to repeat the experiment (so total runs will be cvFolds * repeats)
drop_last = True   # BatchNorm has issues if there are too few samples in the final batch; this drops those offenders
non_CV_div = 1     # used to reduce the number of trials manually if you're not doing CV

SINGLE_IMAGE = '/home/kdobolyi/cellnet/activations/' # used for printing out activations (for a blog post)

# used with calcStats to calculate the mean and stddev of the greyscale images for later normalization; you may need to 
# update this for your dataset
stats_transforms = [#transforms.CenterCrop(224),
                    #transforms.Grayscale(),
                    transforms.Resize([224,224]),
                    transforms.Grayscale(),
                    transforms.ToTensor()]


# obtain the model you want to analyze from the command line
model = eval(sys.argv[2])

# select a GPU number, and select to create a holdout dataset or not, from the command line
# (or skip if you're just calculating pixel stats for a dataset)
if calcStats == False:
    gpu_num = int(sys.argv[1])
    if cuda.is_available():
        torch.cuda.set_device(gpu_num)
        print("starting, using GPU " + str(gpu_num) + "...")
    else:
        device = "cpu"
        print("starting, using CPU")

    # if you want to overrride the default number of epochs specified at the top of this file
    if 'epochs' in model.keys():
        epochs = model['epochs']
    scoring = "_" + str(epochs) + "epochs_lr" + str(learning_rate).replace("0.", "_")            # "_recall", etc
    model['file_label'] = scoring

    # you will set this to True the first time you run this code on a new dataset to generate a global holdout, then 
    # set it to False
    generateHoldout = eval(sys.argv[3])

else:
    # find the mean and stddev for the training data, and quit, so these can be manually copied into the config file
    dataset = datasets.ImageFolder(traindir, transform=transforms.Compose(stat_transforms))
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    getMeanStddev(loader)

# #####################################################################################################################
# DATASET PREP
# #####################################################################################################################

# set up the image transformations to be applied to all models that will be run
image_transforms = {
    'train': transforms.Compose(model['train_transforms']),
    'test': transforms.Compose(model['eval_transforms']),
}

# prepare to store the results from all the runs
results_normal = pandas.DataFrame()
aggregateResults = "./dataframe_" + model['name'] + scoring + ".csv"
if os.path.exists(aggregateResults):
    os.system("rm " + aggregateResults)

def runTrainAndTest(train_dataset, test_dataset, dataloaders, f_label, results_normal):
    """define the basic train and test idiom

    Args:
        train_dataset: ImageFolderWithPaths for the training images; will be used to create a WeightedRandomSampler for 
            the DataLoader
        test_dataset: ImageFolderWithPaths for the testing images
        dataloaders: dict of DataLoader objects we'll modify here in this function
        f_label: a user-chosen label to include in the name of the .torch files generated during training
        results_normal: a dictionary to be updated with the targets and predictions from this test run; collected for 
            voting algorithms later


    Returns:
        confidence_mapping: a zip of image file paths, prediction confidences, target labels, and prediction labels 
            for later reporting and analysis

    """

    # https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/26                                                                        
    # For unbalanced dataset we create a weighted sampler                       
    weights = make_weights_for_balanced_classes(train_dataset.imgs, len(train_dataset.classes))                                                                
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    # Dataloader iterators; test is unique to each CV fold, while the holdout stays the same (above)
    dataloaders['train'] = DataLoader(train_dataset, batch_size=batch_size, sampler = sampler, drop_last=drop_last)
    dataloaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # train and test the model on this CV run
    ctor = model['model']
    print("training model " + str(ctor) + " on fold...")
    neurons = eval(ctor) # we specified the constructor in the config file as a string, so we need to eval it here
    print("Number of traininable parameters: ", neurons.count_parameters())
    neurons.train(dataloaders['train'], None, epochs, device, model, f_label)
    
    print("testing model on fold...")
    neurons.test(dataloaders['test'], device, model, aggregateResults, "test")

    # test the model on the global holdout
    print("testing model on holdout...")
    all_preds, all_targets, confidences, paths = neurons.test(dataloaders['holdout_normal'], device, model, aggregateResults, "holdout")
    results_normal['target'] = all_targets
    results_normal['preds_' + str(f)] = all_preds
    confidence_mapping = zip(paths, confidences, all_preds, all_targets)
    return confidence_mapping

# some models will use CV+holdout, while others will have a static train and test set
if model['usesCV']:
    if generateHoldout: # should only be called once at the start of a new dataset experiment
        makeGlobalHoldout(model)
    holdout_normal = ImageFolderWithPaths(root="HOLDOUT_"+model['traindir'], transform=image_transforms['test'])
    # if we want to evaluate the trained model on just a single image, for blog post purposes
    #holdout_normal = ImageFolderWithPaths(root=SINGLE_IMAGE, transform=image_transforms['test'])   
    dataloaders = {'holdout_normal': DataLoader(holdout_normal, batch_size=batch_size, shuffle=False)}

    # prepare the directory structure and recording for all iterations of this model training on its dataset
    makeCVFolders(model['traindir'], cvFolds, model)

    # run through the number of CV steps, repeating these as specified above
    testIndex = 0
    for r in list(range(repeats)):
        for f in list(range(cvFolds)):
            train_dataset = ImageFolderWithPaths(root="TRAIN_" + str(f), transform=image_transforms['train'])
            test_dataset = ImageFolderWithPaths(root="TEST_" + str(f), transform=image_transforms['test'])
            holdout_confidence_mapping = runTrainAndTest(train_dataset, test_dataset, dataloaders, testIndex, results_normal)
            testIndex += 1
            #break
        #if testIndex == 1:     # these three commented out lines are just used for debugging to avoid doing CVfolds
            #break
else:
    holdout_normal =ImageFolderWithPaths(root=model['holdout'], transform=image_transforms['test'])
    dataloaders = {'holdout_normal': DataLoader(holdout_normal, batch_size=batch_size, shuffle=False)}

    for r in list(range(int((cvFolds * repeats) / non_CV_div))):
        traindir = makeVennDataTraindir(model)
        train_dataset = ImageFolderWithPaths(root=traindir, transform=image_transforms['train'])
        test_dataset = ImageFolderWithPaths(root=model['testdir'], transform=image_transforms['test'])
        holdout_confidence_mapping = runTrainAndTest(train_dataset, test_dataset, dataloaders, r, results_normal)

# calculate voting algorithm results, and write all results to a file
scoreResults(results_normal, 'normal', model, aggregateResults, holdout_confidence_mapping)

if 'venn_data' in model.keys():
    print("Original length of traindir: ", model['original_length'])
    print("VennData length of traindir: ", model['venn_length'])






