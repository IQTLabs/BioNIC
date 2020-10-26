# run these unit tests with the following command:
#
# 		python3 -m unittest dataset_prep_tests.py
#

import unittest
from PIL import Image, ImageDraw

from dataset_prep import *

class TestDataSplit(unittest.TestCase):
	# make sure the holdout images are never in training images
	def test_holdout(self):
		# make folder of raw images
		os.system("mkdir test_images")
		os.system("mkdir test_images/good")
		os.system("mkdir test_images/bad")
		FILE_COUNT = 100
		for i in list(range(FILE_COUNT)):
			img = Image.new('RGB', (100, 30), color = (73, 109, 137))
			d = ImageDraw.Draw(img)
			d.text((10,10), "Image " + str(i), fill=(255,255,0))
			if i < 50:
				img.save('./test_images/good/dummy_img_' + str(i) + '.png')
			else:
				img.save('./test_images/bad/dummy_img_' + str(i) + '.png')


		files = clean(os.listdir('test_images/good'))
		self.assertEqual(len(files),FILE_COUNT // 2)
		files = clean(os.listdir('test_images/bad'))
		self.assertEqual(len(files),FILE_COUNT // 2)

		# call splitter for creating holdout
		model_options = {}
		model_options['traindir'] = 'test_images'
		model_options['labels'] = clean(os.listdir(model_options['traindir']))
		makeGlobalHoldout(model_options)

		# call splitter for CV folders
		model_options['max_class_samples'] = None
		CVFOLDS = 5
		makeCVFolders(model_options['traindir'], CVFOLDS, model_options)

		holdout_files_good = set(clean(os.listdir("./HOLDOUT_" + model_options['traindir'] + "/good")))
		holdout_files_bad = set(clean(os.listdir("./HOLDOUT_" + model_options['traindir'] + "/bad")))
		self.assertEqual(len(holdout_files_good) + len(holdout_files_bad), FILE_COUNT * 0.1)

		for i in list(range(CVFOLDS)):

			# check that no image in the holdout is in the training data
			train_files_good = set(clean(os.listdir("./TRAIN_" + str(i) + "/good")))
			train_files_bad = set(clean(os.listdir("./TRAIN_" + str(i) + "/bad")))
			self.assertEqual(holdout_files_good.isdisjoint(train_files_good), True)
			self.assertEqual(holdout_files_good.isdisjoint(train_files_bad), True)
			self.assertEqual(holdout_files_bad.isdisjoint(train_files_good), True)
			self.assertEqual(holdout_files_bad.isdisjoint(train_files_bad), True)

			# check that no image in the holdout is in the testing data
			test_files_good = set(clean(os.listdir("./TEST_" + str(i) + "/good")))
			test_files_bad = set(clean(os.listdir("./TEST_" + str(i) + "/bad")))
			self.assertEqual(holdout_files_good.isdisjoint(test_files_good), True)
			self.assertEqual(holdout_files_good.isdisjoint(test_files_bad), True)
			self.assertEqual(holdout_files_bad.isdisjoint(test_files_good), True)
			self.assertEqual(holdout_files_bad.isdisjoint(test_files_bad), True)

			# check that no image in the test folder is in the training data
			self.assertEqual(test_files_good.isdisjoint(train_files_good), True)
			self.assertEqual(test_files_bad.isdisjoint(train_files_bad), True)
			self.assertEqual(test_files_good.isdisjoint(train_files_bad), True)
			self.assertEqual(test_files_bad.isdisjoint(train_files_good), True)

			os.system("rm -r TRAIN_" + str(i))
			os.system("rm -r TEST_" + str(i))

		os.system("rm -r " + model_options['traindir'] )
		os.system("rm -r HOLDOUT_" + model_options['traindir'])

class TestScoring(unittest.TestCase):
	# make sure the classes are correctly balanced when measuring accuracy
	def test_weighted_accuracy(self):
		preds = 	[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
		targets = 	[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
		self.assertAlmostEqual(weighted_accuracy(preds, targets), 0)

		preds = 	[0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
		targets = 	[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
		self.assertAlmostEqual(weighted_accuracy(preds, targets), 0.4)

		preds = 	[1, 1, 1, 1, 1, 0, 0, 1, 1, 1]
		targets = 	[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
		self.assertAlmostEqual(weighted_accuracy(preds, targets), 0.7)

		preds = 	[1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
		targets = 	[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
		self.assertAlmostEqual(weighted_accuracy(preds, targets), 0.6)

		preds = 	[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		targets = 	[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
		self.assertAlmostEqual(weighted_accuracy(preds, targets), 0)

		preds = 	[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
		targets = 	[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
		self.assertAlmostEqual(weighted_accuracy(preds, targets), 0.4)

		preds = 	[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
		targets = 	[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
		self.assertAlmostEqual(weighted_accuracy(preds, targets), 0.7)

		preds = 	[1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
		targets = 	[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
		self.assertAlmostEqual(weighted_accuracy(preds, targets), 0.6)

		preds = 	[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
		targets = 	[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
		self.assertAlmostEqual(weighted_accuracy(preds, targets), 1)

 


 
