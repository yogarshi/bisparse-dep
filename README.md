# bisparse-dep

This paper contains data and code from the NAACL 2018 paper ["Robust Cross-lingual Hypernymy Detection using Dependency Context"](https://arxiv.org/abs/1803.11291).

# Data

The `data` directory contains two sub-directories :

- `hyper-hypo` : This dataset contains crowd-sourced hypernyms as positive examples, and crowd-sourced hyponymys as negative examples. This dataset has been used to generate results in Table 3a in the paper.
- `hyper-cohypo` : This dataset contains crowd-sourced hypernyms as positive examples, and automatically extracted co-hpyonyms as negative examples. This dataset has been used to generate results in Table 3b in the paper.

Both directories contain the exact tune/test split used in the paper, for each of four languages - Arabic (ar), French (fr), Russian (ru), and Chinese (zh). Additionally, `hyper-hypo` contains all examples that were crowdsourced - this is a superset of the tune/test data, and contains additional negative examples.

# Pre-trained vectors

The `pre-trained-vecs` directory contains the sparse, bilingual word vectors that have been used to generate the results in the paper. There are 2 vectors per langauge pair (ar-en, fr-en, ru-en, zh-en), per model (window, dependency, joint, delex, unlab), per dataset (hyper-hypo, hyper-cohypo), making for a total of 80 files. They have been organized by dataset, with each dataset folder containing 40 files

Additionally, each dataset folder also contains `hyperparams.txt` which gives the hyperparameters used to generate the vecctors and obtain the results.

# Scripts
- `balAPinc_multi_test.py` - Given a list of cross-lingual word pairs, and two cross-lingual word vector files (one per language), generate balAPinc scores for the word pairs
    - Syntax : `python scripts/balAPinc_multi_test.py <en-word-vectors> <non-en-word-vectors> <word-pairs-file> 0 <balAPinc-parameter> --prefix <optional prefix for output file> `, where
        -  `<en-word-vectors>` ::= File containing word vectors for English
        -  `<non-en-word-vectors>` ::= File containing word vectors for the other language
        -  `<word-pairs-file>` ::= List of (non-English, English) word pairs, with gold label (1 = English word is a hypernym of the non-English word, 0 = otherwise)
        -  `<balAPinc-parameter>` ::= How many features to include while calculating balAPinc? (Integer between 0 and 100, inclusive)
    -  Output : Input file, with a balAPinc score appended at the end of each line
    -  Example usage : `python scripts/balAPinc_multi_test.py pre-trained-vecs/hyper-hypo/ar-en.en.dep_1000.txt.gz pre-trained-vecs/hyper-hypo/ar-en.ar.dep_1000.txt.gz data/hyper-hypo/ar_tune.txt 0 100`
-  ` balAPinc_classification.py` - Given tune and test files, generate classification scores
    -  Syntax : `python scripts/balAPinc_classification.py --training <tune-word-file-with-scores> --test <test-word-file-with-scores>`, where
        - `<tune-word-file-with-scores>` ::= Output of `balAPinc_multi_test.py` when run on tuning data
        - `<test-word-file-with-scores>` ::= Output of `balAPinc_multi_test.py` when run on test data
- `generate_results.sh` - Run this to generate the results reported in the paper (currently generates all BiSparse-Dep (Full, Joint, Delex, Unlabeled) results in Tables 3a, 3b, and 4 )

Scripts to train vectors will be available soon. For now, you can use the scripts from our [prior work](https://github.com/yogarshi/bisparse) if needed.

# References

If you use the code, data, or other resources from this paper, please cite our papers.
Feel free to email us (`yogarshi@cs.umd.edu`, `shyamupa@seas.upenn.edu`) for any questions or assistance.

```
@InProceedings{UpadhyayVyasCarpuatRoth2018,
	author = 	"Upadhyay, Shyam
	and	Vyas, Yogarshi
	and Carpuat, Marine
	and Roth, Dan",
	title = 	"Robust Cross-lingual Hypernymy Detection using Dependency Context",
	booktitle = 	"Proceedings of the 2018 Conference of the North American Chapter of the      Association for Computational Linguistics: Human Language Technologies ",
	year = 	"2018",
	publisher = 	"Association for Computational Linguistics",
	location = 	"New Orleans, Louisiana",
	url = 	"https://arxiv.org/pdf/1803.11291.pdf"
}

@InProceedings{VyasCarpuat2016,
	author = 	"Vyas, Yogarshi
	and Carpuat, Marine",
	title = 	"Sparse Bilingual Word Representations for Cross-lingual Lexical Entailment",
	booktitle = 	"Proceedings of the 2016 Conference of the North American Chapter of the      Association for Computational Linguistics: Human Language Technologies    ",
	year = 	"2016",
	publisher = 	"Association for Computational Linguistics",
	pages = 	"1187--1197",
	location = 	"San Diego, California",
	doi = 	"10.18653/v1/N16-1142",
	url = 	"http://www.aclweb.org/anthology/N16-1142"
}

```
