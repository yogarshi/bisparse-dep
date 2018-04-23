__author__ = 'yogarshi'

import sys
import os
import math
import numpy as np
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mono_nnse_analysis import load_vectors, read_word_file, vector_reps, get_top_words
#from scipy.spatial.distance import cosine
"""
Implementation of the balAPinc score as explained in [1]. The function names exactly correspond to the ones used in [1]

[1] Turney, P., & Mohammad, S. (2013). Experiments with three approaches to recognizing lexical entailment.
Natural Language Engineering, 1(1), 1-42. http://doi.org/10.1017/S1351324913000387
"""

feat_list = []
count = 0

def inc(r,f_u, f_v):
	"""
	:param r: Rank
	:param f_u: Ranked feature list for word u
	:param f_v: Ranked feature list for word v
	"""
	global count
	global feat_list

	inc = 0
	#print r
	for i in range(r):
		if f_u[i] in f_v:
			if f_u[i] not in feat_list[count]:
				feat_list[count].append(f_u[i])
			#print f_u[i], "\t",
			inc += 1
	return inc


def p(r, f_u, f_v):
	"""
	:param r: Rank
	:param f_u: Ranked feature list for word u
	:param f_v: Ranked feature list for word v
	"""
	return float(inc(r,f_u,f_v))/r

def rank(f, f_w):
	"""
	:param f: feature id
	:param f_w: Ranked feature list for word w
	:return:
	"""

	if f in f_w:
		return f_w.index(f)+1
	else:
		return 0


def rel(f,f_w):
	"""
	:param f: feature id
	:param f_w: Ranked feature list for word w
	:return:
	"""
	return 1 - float(rank(f, f_w))/(len(f_w) + 1)



def APinc(f_u, f_v):
	"""
	:param vec1: Numpy array of shape (m,)
	:param vec2: Numpy array of shape (m,)
	:return: APinc(vec1, vec2)
	"""
	sum = 0
	for r in range(1,len(f_u)+1):
		sum += p(r,f_u,f_v)*rel(f_u[r-1], f_v)
	if len(f_u) == 0:
		return 0
	return sum/len(f_u)


def lin(scored_f_u, scored_f_v):
	"""
	:param scored_f_u: Dict of features of word u with feature id as key and score as value
	:param scored_f_v: Dict of features of word v with feature id as key and score as value
	:return:
	"""
	numerator = 0
	denominator = 0
	for each_feat in scored_f_u:
		if each_feat in scored_f_v:
			numerator += scored_f_u[each_feat] + scored_f_v[each_feat]
		denominator += scored_f_u[each_feat]

	for each_feat in scored_f_v:
		denominator += scored_f_v[each_feat]

	if denominator == 0:
		return 0

	return float(numerator)/denominator


def balAPinc(f_u, scored_f_u, f_v, scored_f_v):
	#print f_u
	#print f_v
	#print
	return math.sqrt(APinc(f_u, f_v) * lin(scored_f_u, scored_f_v))



def main(argv):

	global count
	global feat_list

	threshold_hyp =0.05
	num_features_hyp = 20

	#print threshold_hyp, num_features_hyp
	parser = argparse.ArgumentParser(description="Analysis of lexical entailment dataset ")

	parser.add_argument("vector_file_fr", help='The file from which sparse vectors have to be loaded')
	parser.add_argument("vector_file_en", help='The file from which sparse vectors have to be loaded')
	parser.add_argument("word_pairs_file", help='File containing list of word pairs with class')
	parser.add_argument("threshold_hyp", help='Threshold hyperparameter')
	parser.add_argument("num_features_hyp", help='Num features hyperparameter')
	parser.add_argument("-p", "--prefix", help='Optional prefix for output file names', default="log")
	#parser.add_argument("word_pairs_file_neg", help='File containing list of negative examples')

	args = parser.parse_args(args=argv)
	cos = True

	#threshold_hyp =float(args.threshold_hyp)
	#num_features_hyp = int(args.num_features_hyp)

	words_e, embeddings_array_e = load_vectors(args.vector_file_en)
	#top_words_e = get_top_words(words_e, embeddings_array_e, 10)

	#words_f, embeddings_array_f = words_e, embeddings_array_e
	words_f, embeddings_array_f = load_vectors(args.vector_file_fr)
	#top_words_f = get_top_words(words_f, embeddings_array_f, 10)

	idxs_p, word_list1_p, word_list2_p, labels_list, uf_word_list1_p, uf_word_list2_p, uf_labels_list = read_word_file(args.word_pairs_file, words_e, words_f)
	#print word_list1_p
	#print word_list2_p
	#idxs_n, word_list1_n, word_list2_n = read_word_file(args.word_pairs_file_neg, words)

	rep1_array_p = vector_reps(word_list1_p, words_e, embeddings_array_e)
	rep2_array_p = vector_reps(word_list2_p, words_f, embeddings_array_f)
	#rep1_array_n, rep2_array_n = vector_reps(word_list1_n, word_list2_n, words, embeddings_array)

	balAPinc_pos = []
	"""
	if cos:
		outf_name = "{0}".format(args.prefix)
		outf = open(outf_name, 'w')
		for i in range(len(rep1_array_p)):
			outf.write("{0}\t{1}\t{2}\t{3}\n".format(word_list1_p[i], word_list2_p[i], labels_list[i],
											1.0 - cosine(rep1_array_p[i], rep2_array_p[i])))
		outf.close()
	"""
	#balAPinc_neg = []

	for num_features_hyp in [int(args.num_features_hyp)]:

		for threshold_hyp in [float(args.threshold_hyp)]:

			f_u_temp_p = [sorted([(i,x[i]) for i in range(len(x)) if x[i] != 0], reverse=True, key= lambda x:x[1])[:num_features_hyp]
						  for x in rep1_array_p]
			f_v_temp_p = [sorted([(i,x[i]) for i in range(len(x)) if x[i] != 0], reverse=True, key= lambda x:x[1])[:num_features_hyp]
						  for x in rep2_array_p]


			f_u_array_p = [[y[0] for y in x if y[1] > threshold_hyp] for x in f_u_temp_p]
			f_v_array_p = [[y[0] for y in x if y[1] > threshold_hyp] for x in f_v_temp_p]
			scored_f_u_array_p = [{each_key[0]:each_key[1] for each_key in x} for x in f_u_temp_p]
			scored_f_v_array_p = [{each_key[0]:each_key[1] for each_key in x} for x in f_v_temp_p]


			#f_u_temp_n = [ sorted([(i,x[i]) for i in range(len(x)) if x[i] != 0], reverse=True, key= lambda x:x[1])
			#			   for x in rep1_array_n]
			#f_v_temp_n = [ sorted([(i,x[i]) for i in range(len(x)) if x[i] != 0], reverse=True, key= lambda x:x[1])
			#			   for x in rep2_array_n]


			#f_u_array_n = [[y[0] for y in x] for x in f_u_temp_n]
			#f_v_array_n = [[y[0] for y in x] for x in f_v_temp_n]
			#scored_f_u_array_n = [{each_key[0]:each_key[1] for each_key in x} for x in f_u_temp_n]
			#scored_f_v_array_n = [{each_key[0]:each_key[1] for each_key in x} for x in f_v_temp_n]


			balAPinc_array_p = []
			#balAPinc_array_n = []

			a_in_top = 0
			b_in_top = 0
			avg_len = []

			avg_top_score_a = 0
			avg_top_score_b = 0
			top_a_is_top_common = 0
			outf_name = "{2}_{0}_{1}".format(num_features_hyp, threshold_hyp, args.prefix)
			outf = open(outf_name, 'w')
			for i in range(len(word_list1_p)):
				w1 = word_list1_p[i]
				w2 = word_list2_p[i]

				feat_list.append([])
				balAPinc_array_p.append(balAPinc(f_u_array_p[i], scored_f_u_array_p[i], f_v_array_p[i], scored_f_v_array_p[i]))
				outf.write("{0}\t{1}\t{2}\t{3}\n".format(word_list1_p[i], word_list2_p[i], labels_list[i], balAPinc_array_p[i]))
			for i in range(len(uf_word_list1_p)):
				w1 = uf_word_list1_p[i]
				w2 = uf_word_list2_p[i]
				outf.write("{0}\t{1}\t{2}\t{3}\n".format(uf_word_list1_p[i], uf_word_list2_p[i], uf_labels_list[i], 0))
	"""
				print
				for c in range(min(10, len(feat_list[count]) )):
					print top_words_e[feat_list[count][c]]
					print top_words_f[feat_list[count][c]]
					print scored_f_u_array_p[i][feat_list[count][c]], scored_f_v_array_p[i][feat_list[count][c]]

				count += 1
				print "#######################\n"
			outf.close()
	"""

	#print avg_top_score_a/count, avg_top_score_b/count
	#print np.median(avg_len), np.mean(avg_len), np.std(avg_len)
	#print a_in_top, b_in_top, top_a_is_top_common


	#for i in range(len(word_list1_n)):
	#	balAPinc_array_n.append(balAPinc(f_u_array_n[i], scored_f_u_array_n[i], f_v_array_n[i],  scored_f_v_array_n[i]))

	#for i in range(len(word_list1_p)):
	#	print word_list1_p[i], "\t", word_list2_p[i], "\t",  balAPinc_array_p[i]


	##Set up classification task

if __name__ == "__main__":
	main(sys.argv[1:])
