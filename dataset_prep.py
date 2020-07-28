# #########################################################################################################################################################
'''

This module provides helper functions for slicing the incoming dataset into train, test, and global holdout.
It also helps manage the reporting on these different evaluation options.

Typically, you will be running your evaluation on some combination of:
- Train-test split (is this random from a single folder, or will you have your own train and test folders that are different raw images?)
- Datatset augmentation options chosen for a particular model training
- Will you be doing cross-validation with a global holdout, or not? How many folds?
- How many times do you want to train and test a model, to measure stability?

Model performance is evaluated on the weighted accuracy for all labels (so takes class imabalance into account, giving equal weight to all classes in 
the test set). You may want to change this reporting, depending on your ultimate use case for your model; are false negatives worse than false 
positives, etc.

'''
# #########################################################################################################################################################

# PyTorch
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn

# data science tools
import numpy 
import pandas
import os
import random
from datetime import datetime
import statistics
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, f1_score, recall_score, roc_auc_score

DATA_WEIGHT_THRESH = 0.0 # used to evaluate a input purification scheme, VennData (optional)

# #########################################################################################################################################################
# REPORTNG of model results
# #########################################################################################################################################################

# helper function to calculate the accuracy for each label, which is then averaged across all labels. This function takes class imbalance into account,
# in that it gives equal weight to every label.
def weighted_accuracy(preds, targets):

    # if the model always predicts the same thing, give it a score of zero
    if len(set(preds)) == 1:
        return 0

    #return recall_score(preds, targets, average='weighted')

    corrects = {}
    counts = {}
    ctr = 0
    while ctr < len(preds):
        if targets[ctr] not in corrects.keys():
            corrects[targets[ctr]] = 0
        if targets[ctr] not in counts.keys():  
            counts[targets[ctr]] = 0
        if preds[ctr] == targets[ctr]:
            corrects[targets[ctr]] += 1
        counts[targets[ctr]] += 1
        ctr += 1

    accuracies = []
    for k in counts.keys():
        accuracies.append(corrects[k] * 1.0 / counts[k])

    return sum(accuracies) * 1.0 / len(accuracies)

''' Calculates and records the score of a voting ensemble model of all the individual model predictions in results_df;
    appends the score to aggregateResults.csv

    Useful for cases where you are doing multiple runs of CV with individual and a global holdout test set for the same model,
    but trained and tested on different subsets of the dataset (exclusive of the global holdout).

    Writes the following files:
    - results{modelName}_{traindir}.csv : all results on all CV sets for that model+traindir
    - {aggregateResults}.txt.result : all results on all CV sets for that model+traindir, plus voting ensemble score

    Arguments:
    - results_df: a df which stores the target labels, and the predictions of the individual CV models, for the global holdout
    - name: whatever meaningful string you named your model with
    - model_options: the dict of model options, such as traindir and model class type, etc.
    - aggregateResults: the name for the {aggregateResults}.csv file where we will append the voting results
'''
def scoreResults(results_df, name, model_options, aggregateResults, holdout_confidence_mapping, write_file=True):
    # save a copy of the individual model results to a csv for logging
    filename = model_options['name']
    results_df.to_csv("./results" + filename + ".csv")
    file = open("./results" + filename + ".csv")
    data = file.readlines()[1:]
    file.close()

    # using a voting system to find the mode from all the individial model predictions in results  
    all_targets = results_df['target']
    votes = []
    for line in data:
        pieces = eval("[" + line + "]")[1:]
        try:
            vote = statistics.mode(pieces)
        except:
            count = {}
            for p in pieces:
                if p not in count.keys():
                    count[p] = 0
                count[p] += 1
            maxV = -1
            maxK = -1
            for k in count.keys():
                if count[k] > maxV:
                    maxV = count[k]
                    maxK = k
            vote = k
        votes.append(vote)


    # calculate and optionally print out the metrics of your choice
    weighted = weighted_accuracy(votes, all_targets)
    '''if len(set(votes)) != 1:
        weighted = recall_score(votes, all_targets, average='macro')
        #print("test accuracy: " + str(num_correct * 1.0 / num_examples))
        #print('weighted accuracy: ', balanced_accuracy_score(all_preds, all_targets))
        #print('weighted precision: ', precision_score(all_preds, all_targets, average='macro'))
        #print('weighted f1: ', f1_score(all_preds, all_targets, average='macro'))
    else:
        #print(e)
        print("Only one class present in y_true. Recording 0.0 for performance.")
        weighted = 0.0        
    print('OVERALL macro recall: ', weighted)'''

    # calculate various metrics for the voting algorithm
    print('OVERALL ' + name + ' weighted accuracy: ', weighted)
    #print('OVERALL ' + name + ' weighted recall: ', balanced_accuracy_score(votes, all_targets, adjusted=True))
    #print('OVERALL ' + name + ' precision_score: ', precision_score(votes, all_targets, average='weighted'))
    #print('OVERALL ' + name + ' recall: ', recall_score(votes, all_targets, average='weighted'))
    #print('OVERALL ' + name + ' f1: ', f1_score(votes, all_targets, average='macro'))

    if write_file:
        # append the voting model scores to the aggregate results for this model CV experiment
        file = open(aggregateResults, "a+")
        file.write(model_options['traindir'] + "," + model_options['name'] + "_voting,holdout," + str(weighted) + "," + str(datetime.now()) + "\n")
        file.close()

        file = open(aggregateResults+".holdout_confidences.txt", "w")
        for i in holdout_confidence_mapping:
            file.write(str(i)+"\n")
        file.close()


# #########################################################################################################################################################
# SLICE AND DICE THE INCOMING DATASET into an appropriate train-and-test split
# #########################################################################################################################################################

# removes hidden files (starting with .) from a list of filenames (filenames_list)
def clean(filenames_list):
    result = []
    for l in filenames_list:
        if l[0] != '.':
            result.append(l)
    return result

''' Collects inputs that are above a threshold for VennData in cases when we don't use cross-validation (cross-validation datasets are handled in makeCVFolders 
    for VennData).

    VennData is a weighting scheme for the training inputs, calculated elsewhere.

    Takes as argument the model_options dictionary.
'''
def makeVennDataTraindir(model_options):
    # if we're not using VennData, just return the normal, unweighted images in the training directory
    if 'venn_data' not in model_options.keys():
        traindir = model_options['traindir']
    else:
        data_weights = open(model_options['venn_data'])
        lines = data_weights.readlines()[1:]
        data_weights.close()
        labels = model_options['labels']
        if os.path.exists(model_options['traindir'] + "_venn_data"):
            os.system("rm -r " + model_options['traindir'] + "_venn_data")
        os.system("mkdir " + model_options['traindir'] + "_venn_data/")
        ctr = 0
        for label in clean(labels):
            os.system("mkdir " + model_options['traindir'] + "_venn_data/" + label)
            for line in lines:
                file, weight = line.split(",")
                _, label_local, image = file.split('/')
                weight = float(weight)
                if weight >= DATA_WEIGHT_THRESH:
                    if label_local == label:
                        os.system("cp " + model_options['traindir'] + "/" + label + "/" + image + " " + model_options['traindir'] + "_venn_data/" + label + "/" + image)
                        ctr += 1
        data_weights.close()
        traindir = model_options['traindir'] + "_venn_data"
        print("Using VennData weight labeling")

        ctr = 0
        for label in clean(model_options['labels']):
            ctr += len(os.listdir(model_options['traindir'] + "_venn_data/" +  label))
        print("VennData length of " + model_options['traindir'] + " training dataset:", ctr)
        model_options['venn_length'] = ctr
    return traindir

''' For a given directory, creates a global holdout test set that will not be used in any train nor test fold of CV for a model setup in makeCVFolders().
    Creates sub-folders for every individual class/label, to be used with torchvision.datasets.ImageFolder

    Useful to get an idea of generalization, model stability, and needed to evaluate a voting ensemble model. Note, we don't bother with
    a validation set (just train, test, and global holdout) when we are running these experiments, as we are not fine-tuning any model.

    Takes as argument the model_options dictionary.
'''
def makeGlobalHoldout(model_options):
    # prepare the directory structure for the global holdout
    traindir = model_options['traindir']
    testdir_normal = "./HOLDOUT_" + model_options['traindir']

    if os.path.exists(testdir_normal):
        os.system("rm -r " + testdir_normal)
    os.system("mkdir " + testdir_normal)

    # goes through all the possible classes (labels) and randomly splits each class into build vs holdout datasets, writing to disk
    labels = model_options['labels']
    for label in clean(labels):
        files = clean(os.listdir(traindir + "/" + label))
        total = len(files)
        print('total of this label', total)
        indicies = list(range(total))
        random.shuffle(indicies)
        train_i = int(total * 0.1)
        test = indicies[:train_i]       
        
        os.mkdir(testdir_normal + "/" + label)
        for i in test:
            file = files[i]
            print(file)
            os.system("cp ./" + traindir + "/" + label + "/" +  file + " " + testdir_normal + "/" + label + "/" + file)

''' Divides a directory into N cross-validation folds of equal size, and then creates N train-and-test directory pairs of images.
    Creates sub-folders for every individual class/label, to be used with torchvision.datasets.ImageFolder.
    Assumes you have already used the function above to create a global holdout.

    Useful for cases where you are doing multiple runs of CV with individual and a global holdout test set for the same model,
    but trained and tested on different subsets of the dataset (exclusive of the global holdout).

    Arguments:
    - datadir: the path to the directory to split into folds
    - numFolds: how many folds you want for cross-validation
    - model_options: dictionary of configs for the model, including what the possible labels are and whether or not you want to use VennData
      to train on a purified input dataset
'''
def makeCVFolders(datadir, numFolds, model_options):
    # prepare the directories for the folds
    for i in list(range(numFolds)):
        if os.path.exists('TEST_' + str(i)):
            os.system("rm -r TEST_" + str(i))
            os.system("rm -r TRAIN_" + str(i))
    for f in list(range(numFolds)):
        os.system("mkdir TEST_" + str(f))
        os.system("mkdir TRAIN_" + str(f))

    # go through each sub-folder for each type of class, and create it in the TRAIN and TEST dirs
    for label in clean(model_options['labels']):
        print("making folds for " + datadir + "/" + label)
        files_dirty = clean(os.listdir(datadir + "/" + label))
        holdout_files = clean(os.listdir("HOLDOUT_" + datadir + "/" + label))
        files = []
        for f in files_dirty:
            if f not in holdout_files:
                files.append(f)

        # randomly select the files for the folds, using a cap as requested
        indicies = list(range(len(files)))
        random.shuffle(indicies)
        if model_options['max_class_samples'] != None:
            indicies = indicies[:model_options['max_class_samples']]
        slice_size = int(len(indicies) / numFolds)
        folds = [indicies[i * slice_size:(i + 1) * slice_size] for i in range((len(indicies) + slice_size - 1) // slice_size )]  
        if len(folds) > numFolds:
            folds[-2].extend(folds[-1])
            folds = folds[:-1]

        # check to see if we're using VennData weights
        if 'venn_data' in model_options.keys():
            keep_files = []
            data_weights_file = open(model_options['venn_data'])
            lines = data_weights_file.readlines()
            data_weights_file.close()
            for line in lines[1:]:
                file, weight = line.split(",")
                weight = float(weight)
                if weight >= DATA_WEIGHT_THRESH:
                    keep_files.append(file)

        # create the test and train folds
        for fold in list(range(numFolds)):
            os.system("mkdir TEST_" + str(fold) + "/" + label)
            os.system("mkdir TRAIN_" + str(fold) + "/" + label)
            train_idx = []
            test_idx = []
            for i in list(range(numFolds)):
                if i == fold:
                    test_idx = folds[i]
                else:
                    train_idx.extend(folds[i])

            for i in train_idx:
                if 'venn_data' in model_options.keys():
                    if (datadir + "/" + label + "/" + files[i]) in keep_files:
                        os.system("cp ./" + datadir + "/" + label + "/" + files[i] + " ./TRAIN_" + str(fold) + "/" + label + "/" + files[i])
                else:
                    os.system("cp ./" + datadir + "/" + label + "/" + files[i] + " ./TRAIN_" + str(fold) + "/" + label + "/" + files[i])
            for i in test_idx:
                os.system("cp ./" + datadir + "/" + label + "/" + files[i] + " ./TEST_" + str(fold) + "/" + label + "/" + files[i])

    if 'venn_data' in model_options.keys():
        print("Using VennData weight labeling")
        ctr = 0
        for label in clean(model_options['labels']):
            ctr += len(os.listdir('TRAIN_0/' + label))
        print("VennData length of " + datadir + " training dataset:", ctr)
        model_options['venn_length'] = ctr


# Calculates weights to be passed to torch.utils.data.sampler.WeightedRandomSampler when training to help with class imbalance issues
def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight  

# allow us to pass the name of the image in to the dataloader, so we can pull these when calculating confidence scores
class ImageFolderWithPaths(datasets.ImageFolder):
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
