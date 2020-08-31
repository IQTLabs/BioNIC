# #####################################################################################################################
'''

This file is a scratchpad of code used for preparing the folders of the dataset, including:

 - Inverting a greyscale image, so that all cells are on a black background
 - Pulling a random subset of images into a holdout group


'''
# #####################################################################################################################

# Data science tools
import os
import sys
from PIL import Image, ImageOps
import numpy as np
import image_slicer
import random
import torchvision.transforms.functional as F
import pandas

# local libraries
from dataset_prep import clean

class Invert(object):	
	""" Inverting a greyscale image """

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
	""" Inverts all the images in the specified directory (so you can change cells to be on a black background, 
		instead of white background, for example) """
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

def createRandomSubset(root, size):
	""" Places random files into a new folder, <root>_subset_<size> 

		Arguments:
			root: the source folder of all the images, which has a subfolder for each label (like ImageFolder)
			size: what the size of the random subset should be

	"""

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

def createTiles(root, dest, subfolders, centerCropSize, numTiles):
	""" slices a folder (assumed to be all of the same label) into tiles, called <root>_tiles

		Arguments:
			root: the source folder of all the images, which has a subfolder for each label (absolute path)
			subfolders: list of names of subfolders you want to tile (relative path folder name only)
			centerCropSize: to what size should the raw image be center-cropped to (or None)
			numTiles: how many tiles you want to create out of the (cropped) raw image

	"""

	if os.path.exists(dest):
		os.system("rm -r " + dest)
	os.system("mkdir " + dest)

	# process all the subfolders
	for label in subfolders:
		os.system("mkdir " + dest + "/" + label)

		# process individual image
		files = os.listdir(root + "/" + label)
		for f in files:

			if f.startswith('.'):
				continue

			# center-crop the image (or not) and create and save tiles
			if centerCropSize != None:

				# crop the image
				im = Image.open(root + "/" + label + "/" + f)
				width, height = im.size  
				left = (width - centerCropSize)/2
				top = (height - centerCropSize)/2
				right = (width + centerCropSize)/2
				bottom = (height + centerCropSize)/2
				im = im.crop((left, top, right, bottom))
				im.save(dest + "/" + label + "/cropped_" + f)

				tiles = image_slicer.slice(dest + "/" + label + "/cropped_" + f, numTiles, save=False)
				os.system("rm " + dest + "/" + label + "/cropped_" + f)
			else:
				tiles = image_slicer.slice(root + "/" + label + "/" + f, numTiles, save=False)	
			
			image_slicer.save_tiles(tiles, directory= dest + "/" + label + "/", prefix='tile_' + f)
			print("finished tiling " + label + " to " + dest)

createTiles("/Users/kdobolyi/Downloads/Labeled_Images_color", "/Users/kdobolyi/Downloads/Labeled_Images_color_tiles", 
	['good', 'bad'], 1200, 2)

def makeImageFolders(csv, file_dir_source, file_dir_dest, labels):
	""" takes a directory of raw images, and a csv in Image_Name,label format, and turns it into ImageFolder style

		Arguments:
			csv: the csv with the file names and labels. Must have columns called "Image_Name" and "label". Image
			 	names should be relative paths to file_dir_source.
			file_dir_source: the directory containing all the images references by your CSV (absolute path)
			file_dir_dest: the root where you want it to create a subfolder for each label (absolute path)
			labels: list of labels you want to use (should match labels in your csv)

	"""
	key = pandas.read_csv(csv)
	print(key['label'].value_counts())

	# overwrites any old dest files, making new folders
	if os.path.exists(file_dir_dest):
		os.system("rm -r " + file_dir_dest)
	os.system("mkdir " + file_dir_dest)
	for i in labels:
		os.system("mkdir " + file_dir_dest + "/" + i)

	# makes a copy of all images in the correct folder
	def processRow(img, label, file_dir_source, file_dir_dest):
		os.system("cp " + file_dir_source + "/" + img + " " + file_dir_dest + "/" + label + "/" + img)
	key[['Image_Name', "label"]].apply(lambda x: processRow(*x, file_dir_source, file_dir_dest), axis=1) 
	print("fininshed splitting raw image folder into ImageFolder style folder at " + file_dir_dest)

#makeImageFolders('/Users/kdobolyi/Downloads/Labeled_Images_2ndSet/Image_Label_association.csv', 
#	'/Users/kdobolyi/Downloads/Labeled_Images_color_split', '/Users/kdobolyi/Downloads/Labeled_Images_color/', ['good', 'bad'])
