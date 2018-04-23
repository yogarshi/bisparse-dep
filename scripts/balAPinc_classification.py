__author__ = 'yogarshi'

import sys, os
import numpy as np
import argparse
from random import sample, shuffle, seed
from sklearn.metrics import f1_score,precision_recall_fscore_support,accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold

def get_f_score(tp,tn,fp,fn,w_0,w_1):

	#print tp, tn, fp, fn

	if tp + fp > 0:
		precision_1 = float(tp) / float(tp + fp)
	else:
		precision_1 = 0

	if tp + fn > 0:
		recall_1 = float(tp) / float(tp + fn)
	else:
		recall_1 = 0

	if precision_1 + recall_1 > 0:
		f_1 = (2 * precision_1 * recall_1) / (precision_1 + recall_1)
	else:
		f_1 = 0



	if tn + fn > 0:
		precision_0 = float(tn) / float(tn + fn)
	else:
		precision_0 = 0

	if tn + fp > 0:
		recall_0 = float(tn) / float(tn + fp)
	else:
		recall_0 = 0

	if precision_0 + recall_0 > 0:
		f_0 = (2 * precision_0 * recall_0) / (precision_0 + recall_0)
	else:
		f_0 = 0

	p = w_0 * precision_0 + w_1 * precision_1
	r = w_0 * recall_0 + w_1 * recall_1
	f = w_0 * f_0 + w_1 * f_1

	accuracy = float(tp + tn)/float(tp+tn+fp+fn)

	#print f

	return p, r, f, accuracy


def score_dataset(dataset, threshold):

	dataset = sorted(dataset, key=lambda x: x[2])
	total_pos = len([x for x in dataset if x[-1] == 1])
	total_neg = len(dataset) - total_pos
	total = total_pos + total_neg

	pos_lte_threshold = 0
	neg_lte_threshold = 0


	w_0 = 0.5
	w_1 = 0.5
	#w_0 = float(total_neg) / float(total)
	#w_1 = float(total_pos) / float(total)

	for i in range(len(dataset)):
		if dataset[i][2] > threshold:
			break
		if dataset[i][-1] == 0:
			neg_lte_threshold += 1
		else:
			pos_lte_threshold += 1

	tp = total_pos - pos_lte_threshold
	tn = neg_lte_threshold
	fp = total_neg - neg_lte_threshold
	fn = pos_lte_threshold

	return get_f_score(tp, tn, fp, fn, w_0, w_1)

def get_threshold(training_set):
	training_set = sorted(training_set, key=lambda x: x[2])
	total_pos = len([x for x in training_set if x[-1] == 1])
	total_neg = len(training_set) - total_pos
	#print total_pos, total_neg
	total = total_pos + total_neg

	best_threshold = -1
	best_score = 0
	pos_lte_threshold = 0
	neg_lte_threshold = 0

	w_0 = 0.5
	w_1 = 0.5
	#w_0 = float(total_neg) / float(total)
	#w_1 = float(total_pos) / float(total)

	#print total_pos, total_neg

	for i in range(len(training_set)):
		if training_set[i][-1] == 0:
			neg_lte_threshold += 1
		else:
			pos_lte_threshold += 1
		#print neg_lte_threshold, pos_lte_threshold
		tp = total_pos - pos_lte_threshold
		tn = neg_lte_threshold
		fp = total_neg - neg_lte_threshold
		fn = pos_lte_threshold

		p, r, f, accuracy = get_f_score(tp, tn, fp, fn, w_0, w_1)

		# score = neg_lte_threshold + total_pos - pos_lte_threshold
		if f >= best_score:
			best_score = f
			best_threshold = training_set[i][2]

	#print "Training Score : ", float(best_score)/len(training_set)

	return best_threshold


def standard_evaluation(training_list, folds,fname_prefix):

	global write

	#print training_list[-10:]

	## Create folds
	test_folds = [[] for i in range(folds)]
	training_folds = []

	# First randomize the training set
	#training_size = int((1.0 1.0/folds) * len(training_list))
	seed(100)
	shuffle(training_list)
	count = 0
	for i in range(len(training_list)):
		test_folds[count].append(training_list[i])
		count += 1
		count %= folds

	#print len(test_folds)
	#for each in training_folds:
	#	print len(each)
	#Now create the test sets
	for i in range(folds):
		fold = []
		for each_item in training_list:
			if each_item not in test_folds[i]:
				fold.append(each_item)
		print len(fold), len(test_folds[i])
		training_folds.append(fold)

	total = 0
	total_accuracy = 0
	#For each fold, get the threshold from the training data
	for i in range(folds):
		threshold = get_threshold(training_folds[i])
		if write:
			fname = fname_prefix + '_' +str(i)
			with open(fname, 'w') as f:
				for each_element in test_folds[i]:
						if each_element[2] > threshold:
							f.write("{0}\t{1}\tTrue\n".format(each_element[0],each_element[1]))
						else:
							f.write("{0}\t{1}\tFalse\n".format(each_element[0],each_element[1]))

		#print threshold
		#Apply threshold to test data
		p, r, f, accuracy = score_dataset(test_folds[i], threshold)
		#print score
		total += f
		print f
		print len(test_folds[i])
		total_accuracy += accuracy

	print total / folds, total_accuracy/folds


def standard_evaluation2(test_set,folds,prefix):
	kf = KFold(n_splits=folds, shuffle=False)

	X = np.ones(len(test_set))
	for train, test in kf.split(X, test_set):
		curr_train = [test_set[x] for x in train]
		curr_test = [test_set[x] for x in test]
		different_evaluation(curr_train,curr_test,prefix)
def different_evaluation(training_set, test_set, prefix):

	# print test_set

	global write
	#Get the threshold from the training set
	training_set = sorted(training_set,key=lambda x:x[2])

	training_gold = [x[3] for x in training_set]
	training_preds = [1]* len(training_gold)

	threshold = 0.0
	best_f = accuracy_score(training_gold,training_preds)
	#best_f = f1_score(training_gold,training_preds,average='binary')
	for i in range(len(training_preds)):
		training_preds[i] = 0
		curr_f = accuracy_score(training_gold,training_preds)
		#curr_f = f1_score(training_gold,training_preds,average='binary')
		if curr_f > best_f:
			threshold = training_set[i][2]
			best_f = curr_f

	test_gold = [x[3] for x in test_set]
	test_preds = [1  if x[2]>=threshold else 0 for x in test_set]


	#print set(test_preds)
	p,r,f,support = precision_recall_fscore_support(test_gold,test_preds,average='binary')
	a = accuracy_score(test_gold,test_preds)
	print p, r, a,f, threshold

	final_test_preds = {}

	count = 0
	for each_example in test_set:
		wp = (each_example[0],each_example[1])
		final_test_preds[wp] = test_preds[count]
		count += 1

	#print len(final_test_preds)
	return final_test_preds



	# if write:
	# 	fname = prefix
	# 	with open(fname, 'w') as f:
	# 		for each_element in test_set:
	# 			j = "True" if each_element[2] > threshold else "False"
	# 			f.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(each_element[0], each_element[1], each_element[2],
	# 			"True" if each_element[3] == 1 else "False", j))

	#Score the test set
	# p, r, f, accuracy = score_dataset(test_set, threshold)
	# #print threshold
	# print threshold, "\t", f, "\t", accuracy



def main(argv):

	global write

	parser = argparse.ArgumentParser(description="Classification of lexical entailment dataset ")

	parser.add_argument("--training", help='The training data')
	parser.add_argument("-f", "--folds", help='Number of CV folds', default=10)
	parser.add_argument("-te", "--test", help='Optional test data', default=None)
	parser.add_argument("-w", "--write", help='Write k-fold output to file', action="store_true",default=False)
	parser.add_argument("-p", "--prefix", help='Prefix if write = true', default="fold_info")
	parser.add_argument("--orig_test",default='test')
	args = parser.parse_args(args=argv)

	write = args.write

	# Load examples in a list of tuples of type (word1, word2, score, class)
	full_list_training = []

	with open(args.training) as f:
		for each_line in f:
			word1, word2, classification, score = each_line.strip().split("\t")
			#print classification.strip()+"h"
			full_list_training.append((word1, word2, float(score),int(classification)))

	#If test data has been supplied, load test data
	if args.test != None:
		full_list_test = []

		with open(args.test) as f:
			for each_line in f:
				word1, word2, classification, score = each_line.strip().split("\t")
				full_list_test.append((word1, word2, float(score), int(classification)))

		#Get threshold on training data
		final_test_preds = different_evaluation(full_list_training, full_list_test, args.prefix)

		# Write the predictions according to context-aware dataset problem
		#with open(args.orig_test) as f_in,open('test_preds','w') as f_out:
		#	for each_line in f_in:
		#		x = each_line.strip().split("\t")
		#		wp = (x[3],x[7])
		#		if wp in final_test_preds:

		#			pred = final_test_preds[wp]

		#			f_out.write(each_line.strip())
		#			f_out.write("\t")
		#			f_out.write(str(pred))
		#			f_out.write("\n")

	#else call CV function
	else:
		standard_evaluation2(full_list_training, int(args.folds), args.prefix)



if __name__ == "__main__":
	main(sys.argv[1:])
