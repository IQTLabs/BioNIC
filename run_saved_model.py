# #####################################################################################################################
'''

This module is used to run a saved model, defined in a config file (such as augment_config.py), and stores its 
predictions and confidences in a csv file for analysis.


'''
# #####################################################################################################################


# PyTorch
from torchvision import datasets, models
import torch
from torch.utils.data import DataLoader, sampler, random_split, ConcatDataset

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

# model config files
from augment_config import *
from cellnet_config import *

# #####################################################################################################################
# BASIC CONFIGURATION
# #####################################################################################################################

CONFIDENCE_THRESH = 0.85 # what you want to set the confidence to, to split for low versus high confidence predictions

batch_size = 32
learning_rate = 1e-4 # just needed to pass to our ctor, it will be ignored because we're not training
device = "cuda"

# choose a GPU from the command line arguments
gpu_num = int(sys.argv[1])
if cuda.is_available():
    torch.cuda.set_device(gpu_num)
print("starting, using GPU " + str(gpu_num) + "...")

# which model you want to run the cross-validation experimental setup on from models_config.py, augment.py
model = eval(sys.argv[2])


# #####################################################################################################################
# RUN A PREVIOUSLY TRAINED MODEL
# #####################################################################################################################

print("testing a saved model...")
neurons = eval(model['model'])

# find the holdout that you want to use to test on (will be specified in the config file for each model), and test it
results_normal = pandas.DataFrame()
counter = 0
while os.path.exists("./" + model['name'] + "_" + str(counter) + ".torch"):
	neurons.model.load_state_dict(torch.load("./" + model['name'] + "_" + str(counter) + ".torch"))
	neurons.model.eval()
	
	if model['holdout'] == False:
		model['holdout'] = model['saved_holdout']
	holdout_normal = ImageFolderWithPaths(root=model['holdout'], transform=transforms.Compose(model['eval_transforms']))
	all_preds, all_targets, confidences, paths = neurons.test(DataLoader(holdout_normal, batch_size=batch_size, shuffle=False), 
		device, model, None, "loaded")
	results_normal['target'] = all_targets
	results_normal['preds_' + str(counter)] = all_preds
	confidence_mapping = zip(paths, confidences, all_preds, all_targets)
	counter += 1
print("------------------------------------------\nRESULTS FROM VOTING ACROSS ALL MODELS:")
scoreResults(results_normal, 'loaded', model, None, confidence_mapping, write_file=False)

# create a csv with the targets, predictions, confidences, and filenames for every image in your holdout test set
c_file = model['name']+"_confidences.csv"
file = open(c_file, "w")
file.write("filename,goal,prediction,confidence,cuda\n")
ctr = 0
while ctr < len(confidences):
	filename = paths[ctr]
	confidence = confidences[ctr]
	pred = all_preds[ctr]
	goal = all_targets[ctr]
	file.write(filename + "," + str(goal) + "," + str(pred) + "," + str(confidence) + "\n")
	ctr += 1
file.close() 
print("wrote predictions and their confidences to " + c_file)

# process the predicitions to get stats on how often high versus low confidence predictions were actuall correct
df_confidence = pandas.read_csv(c_file)
df_confidence['confidence'] = df_confidence['confidence'].apply(lambda x: float(x.split('(')[1]))
df_confidence['matched'] = df_confidence['goal'] == df_confidence['prediction']
df_high = df_confidence[df_confidence['confidence'] >= CONFIDENCE_THRESH]
df_high_yes = df_high[df_high['goal'] == 1]
df_high_bad = df_high[df_high['goal'] == 0]
df_low = df_confidence[df_confidence['confidence'] < CONFIDENCE_THRESH]
df_low_yes = df_low[df_low['goal'] == 1]
df_low_bad = df_low[df_low['goal'] == 0]

# print out the performance of the models, based on confidence in predidctions
if len(df_high_yes) != 0:
	print("Correctly matched high confidence usable: " + str(sum(df_high_yes['matched']) * 1.0 / len(df_high_yes)) + \
	 " out of " + str(len(df_high_yes)))
else:
	print("Correctly matched high confidence usable: N/A out of 0")

if len(df_high_bad) != 0:
	print("Correctly matched high confidence un-usable: " + str(sum(df_high_bad['matched']) * 1.0 / len(df_high_bad)) + \
		" out of " + str(len(df_high_bad)))
else:
	print("Correctly matched high confidence un-usable: N/A out of 0")

if len(df_low_yes) != 0:
	print("Correctly matched low confidence usable: " + str(sum(df_low_yes['matched']) * 1.0 / len(df_low_yes)) + \
		" out of " + str(len(df_low_yes)))
else:
	print("Correctly matched low confidence usable: N/A out of 0")

if len(df_low_bad) != 0:
	print("Correctly matched low confidence un-usable: " + str(sum(df_low_bad['matched']) * 1.0 / len(df_low_bad)) + \
		" out of " + str(len(df_low_bad)))
else:
	print("Correctly matched low confidence un-usable: N/A out of 0")
print("DONE.")


