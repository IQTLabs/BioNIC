# #########################################################################################################################################################
'''

This file is a scratchpad of code used for preparing the folders of the dataset, including:

 - Inverting a greyscale image, so that all cells are on a black background
 - Pulling a random subset of images into a holdout group


'''
# #########################################################################################################################################################

# Data science tools
import os
import sys
from PIL import Image, ImageOps
import numpy as np
import image_slicer
import random
import torchvision.transforms.functional as F

# local libraries
from dataset_prep import clean

# #########################################################################################################################################################
# Inverting a greyscale image
# #########################################################################################################################################################

class Invert(object):	
	def invert(self, img):
		if not F._is_pil_image(img):
			raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

		if img.mode == 'RGBA':
			r, g, b, a = img.split()
			rgb = Image.merge('RGB', (r, g, b))
			inv = ImageOps.invert(rgb)
			r, g, b = inv.split()
			inv = Image.merge('RGBA', (r, g, b, a))
		elif img.mode == 'LA':
			l, a = img.split()
			l = ImageOps.invert(l)
			inv = Image.merge('LA', (l, a))
		else:
			inv = ImageOps.invert(img)
		return inv

	def __call__(self, img):
		return self.invert(img)

	def __repr__(self):
		return self.__class__.__name__ + '()'

def invert(file_dir):
	inv = Invert()
	files = os.listdir(file_dir)

	print("starting conversion...")
	ctr = 0
	for f in files:
		print(ctr)
		ctr += 1
		try:
			gr = Image.open(file_dir + f)
			inverse = inv(gr)
			inverse.save(file_dir + f)
		except:
			print(f)
	print("...finished conversion")

#invert("/home/kdobolyi/cellnet/malaria_cell_images/Parasitized/")

# #########################################################################################################################################################
# getting random files into a folder
# #########################################################################################################################################################
def createRandomSubset(root, size):

	mini = root + "_subset_" + str(size)
	if os.path.exists(mini):
		os.system("rm -r " + mini)
	os.system("mkdir " + mini)

	labels = clean(os.listdir(root))
	# go through each sub-folder for each type of class, and create it in the TRAIN and TEST dirs
	for label in labels:
		files_dirty = clean(os.listdir(root + "/" + label))
		holdout = clean(os.listdir('HOLDOUT_' + root))
		files = []
		for file in files_dirty:
			if file not in holdout:
				files.append(file)

		# randomly select the files for the folds, using a cap as requested
		indicies = list(range(len(files)))
		random.shuffle(indicies)
		indicies = indicies[:size]

		os.system("mkdir " + mini + "/" + label)
		for i in indicies:
			os.system("cp " + root + "/" + label + "/" + files[i] + " " + mini + "/" + label + "/" + files[i])

#createRandomSubset("malaria_cell_images", 500)


