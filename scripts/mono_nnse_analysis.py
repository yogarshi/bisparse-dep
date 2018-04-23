__author__ = 'yogarshi'


import argparse, sys
import math
import gzip
import numpy as np

def load_vectors(vector_file):

	if(vector_file[-3:] == ".gz"):
		f = gzip.open(vector_file)
	else:
		f = open(vector_file)

	#ig = f.readline()
	#ig = f.readline()
	words = []
	embeddings = []
	for each_line in f:
		x = each_line.strip().split()
		words.append(x[0])
		embeddings.append([float(y) for y in x[1:]])
	embeddings_array = np.array(embeddings)
	return words, embeddings_array


def get_top_words(words, embeddings_array, k):

	top_words = []
	embeddings_arrayT = embeddings_array.transpose()
	for each_dimension in embeddings_arrayT:
		#print each_dimension
		top10 = [words[x] for x in each_dimension.argsort()[-k:][::-1]]
		top_words.append(top10)
		#all_words = [words[x] for x in range(len(each_dimension)) if each_dimension[x] > 0]
		#top_words.append(all_words)


	return top_words


def read_word_file(word_file, words_e, words_f):

	f = open(word_file)
	word1_list = []
	word2_list = []
	label_list = []
	idxs = []
	unfound_idxs = []
	unfound_word1_list = []
	unfound_word2_list = []
	unfound_label_list = []
	count = 0
	for each_line in f:
		#print each_line
		x = each_line.strip().split("\t")
		word1 = x[0].split('-')[0].lower()
		word2 = '_'.join(x[1].split('-')).lower()
		label = x[2]
		try:
			ix1 = words_e.index(word1)
			ix2 = words_f.index(word2)
			idxs.append(count)
			count += 1
			word1_list.append(word1)
			word2_list.append(word2)
			label_list.append(label)
		except:
			count += 1
			unfound_word1_list.append(word1)
			unfound_word2_list.append(word2)
			unfound_label_list.append(label)
			continue

	return idxs, word1_list, word2_list, label_list, unfound_word1_list, unfound_word2_list, unfound_label_list


def vector_reps(wl1, words, embeddings_array):

	rep1_array = []

	for i in range(len(wl1)):
		w1 = wl1[i]
		ix1 = words.index(w1)

		rep1_array.append(embeddings_array[ix1])

	return rep1_array


def get_same(rep1_array, rep2_array):

	same_array = []

	for i in range(len(rep1_array)):

		rep1 = rep1_array[i]
		rep2 = rep2_array[i]

		curr_same = [j for j in range(len(rep1)) if (rep1[j]!=0 and rep2[j]!=0)]
		same_array.append(curr_same)

	return same_array

def normalize(l):
	x = sum(l)
	for i in range(len(l)):
		l[i] /= x

	return l


def smooth_normalize(l, alpha=0.0001):
	x = sum(l) +  alpha*len(l)
	for i in range(len(l)):
		l[i] += alpha
		l[i] /= x

	return l


def bidistribute(rep1_array, rep2_array, same_array):

	bidis_rep1_array = []
	bidis_rep2_array = []

	for i in range(len(rep1_array)):
		d1 = []
		d1_prime = []
		d2 = []
		d2_prime = []

		curr_rep1 = rep1_array[i]
		curr_rep2 = rep2_array[i]

		for j in range(len(curr_rep2)):
			if j in same_array[i]:
				d1_prime.append(curr_rep1[j])
				d2_prime.append(curr_rep2[j])
			elif curr_rep1[j] != 0:
				d1.append(curr_rep1[j])
			elif curr_rep2[j] != 0:
				d2.append(curr_rep2[j])

		#normalize
		bidis_rep1_array.append((normalize(d1), normalize(d1_prime)))
		bidis_rep2_array.append((normalize(d2), normalize(d2_prime)))

	return bidis_rep1_array, bidis_rep2_array


def info(ixa, ixb, embeddings_array):
	r_a = embeddings_array[ixa]
	r_b = embeddings_array[ixb]
	nz_a = [x for x in range(len(r_a)) if r_a[x] != 0]
	nz_b = [x for x in range(len(r_b)) if r_b[x] != 0]
	same = [x for x in nz_a if x in nz_b]
	same_r_a = [r_a[x] for x in same]
	same_r_b = [r_b[x] for x in same]
	nz_other_a = [x for x in nz_a if x not in same]
	nz_other_b = [x for x in nz_b if x not in same]
	nz_other_r_a = [r_a[x] for x in nz_other_a]
	nz_other_r_b = [r_b[x] for x in nz_other_b]
	max_5 = same_r_a.index(max(same_r_a))
	max_6 = same_r_b.index(max(same_r_b))
	max_9 = nz_other_a.index(max(nz_other_a))
	max_10 = nz_other_b.index(max(nz_other_b))
	return (r_a, r_b, nz_a, nz_b, same, same_r_a, same_r_b, nz_other_a, nz_other_b, nz_other_r_a, nz_other_r_b,
			max_5, max_6, max_9, max_10)


def main(argv):

	parser = argparse.ArgumentParser(description="Analysis of lexical entailment dataset ")

	parser.add_argument("vector_file", help='The file from which sparse vectors have to be loaded')
	#parser.add_argument('word_pairs_file', help='File containing list of word pairs to analyze')

	args = parser.parse_args()

	#Load the vectors
	words, embeddings_array = load_vectors(args.vector_file)
	top_words = get_top_words(words, embeddings_array, 10)

	avg_top_words_length = 0
	top_words_l = []
	count =0
	for each in top_words:
		print count, "\t", each
		count += 1
		avg_top_words_length += len(each)
		top_words_l.append(len(each))

		#print np.median(top_words_l)
		#print avg_top_words_length/float(count), np.std(top_words_l)

"""
	#Read word_file
	idxs, word_list1, word_list2 = read_word_file(args.word_pairs_file, words)

	#Get vector representations
	rep1_array, rep2_array = vector_reps(word_list1, word_list2, words, embeddings_array)

	#Get common dimensions
	same_array = get_same(rep1_array, rep2_array)

	#Convert each vector to a prob dist
	dis_array1 =  [smooth_normalize(x) for x in rep1_array]
	dis_array2 =  [smooth_normalize(x) for x in rep2_array]


	#Convert each vector rep into two distributions
	#bidis_array1, bidis_array2 = bidistribute(rep1_array, rep2_array, same_array)
	kldiv_array_pq = [entropy(dis_array1[i], dis_array2[i])  for i in range(len(dis_array1))]
	kldiv_array_qp = [entropy(dis_array2[i], dis_array1[i])  for i in range(len(dis_array1))]

	#kldiv_array = [entropy(bidis_array2[i][1], bidis_array1[i][1], base=2) for i in range(len(bidis_array1))
	#			   if not (bidis_array1[i][1] == [] and bidis_array2[i][1] == [])]

	count = 0
	for each in kldiv_array_pq:
		print each - kldiv_array_qp[count], "\t", word_list1[count], "\t", word_list2[count]
		count += 1








	word_file = open("neg_word_list.txt")
	same_array = []
	diff_array = []
	nz_r1_array = []
	nz_r2_array = []
	idxs = []
	rep1_array = []
	rep2_array = []
	word1_list = []
	word2_list = []
	count = 0
	for each_line in word_file:
			x = each_line.strip().split()
			word1 = x[0].split('-')[0]
			word2 = x[1].split('-')[0]
			try:
					ix1 = words.index(word1)
					ix2 = words.index(word2)
					idxs.append(count)
					count += 1
					word1_list.append(word1)
					word2_list.append(word2)

			except:
					count += 1
					continue

			rep1 = embeddings_array[ix1]
			rep2 = embeddings_array[ix2]
			non_zero_rep1 = [x for x in range(300) if rep1[x] != 0]
			non_zero_rep2 = [x for x in range(300) if rep2[x] != 0]
			same = [x for x in non_zero_rep1 if (x in non_zero_rep2)]# and abs(rep1[x]) > 0.1 and abs(rep2[x]) > 0.1)]
			diff = [rep1[i] - rep2[i] for i in same]
			nz_r1_array.append(non_zero_rep1)
			nz_r2_array.append(non_zero_rep2)
			same_array.append(same)
			diff_array.append(diff)


	diff_min_max = sorted([(min(diff_array[i]),max(diff_array[i]), i) for i in range(len(diff_array)) if diff_array[i] != [] ], reverse=True, key = lambda p : p[1])
	y_max = [p[1] for p in diff_min_max]
	y_min = sorted([p[0] for p in diff_min_max])
	#x = [i for i in range(len(y_max))]
	#plt.plot(x,y_max,'r',x,y_min,'b')
	#plt.show()
	y_same = sorted([len(x) for x in same_array])
	plt.plot(range(len(y_same)), y_same)
	y1_overlap = sorted([ float(len(same_array[i]))/len(set(nz_r1_array[i])) for i in range(len(nz_r1_array))])
	y2_overlap = sorted([ float(len(same_array[i]))/len(set(nz_r2_array[i])) for i in range(len(nz_r2_array))])
	#plt.plot(range(len(y1_overlap)), y1_overlap, 'r', range(len(y2_overlap)), y2_overlap, 'b')
	#plt.yscale('log')
	plt.show()
"""

if __name__ == "__main__":
	main(sys.argv[1:])
