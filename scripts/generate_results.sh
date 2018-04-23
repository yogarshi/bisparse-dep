for dataset in hyper-cohypo; do
	rm ${dataset}_results.txt
	for lang in ar fr ru zh; do
		for model in dep_1000 joint delex undep; do
		
		echo $dataset $lang-en $model

		# Grab balapin hyperparameter from the hyper parameter file
		#bp_param=`grep model ../pre-trained-vecs/${dataset}/hyperparams.txt | grep ${lang}-en | cut -d' ' -f6`
		bp_param=`grep $model ../pre-trained-vecs/${dataset}/hyperparams.txt | grep ${lang}-en | cut -d' ' -f6`
		echo $bp_param
		# Generate scores for tuning set
		python balAPinc_multi_test.py ../pre-trained-vecs/${dataset}/${lang}-en.en.${model}.txt.gz ../pre-trained-vecs/${dataset}/${lang}-en.${lang}.${model}.txt.gz ../data/${dataset}/${lang}_tune.txt 0 ${bp_param} --prefix log_tune_${dataset}_${lang}_${model}
		
		# Generate scores for tuning set/	
		python balAPinc_multi_test.py ../pre-trained-vecs/${dataset}/${lang}-en.en.${model}.txt.gz ../pre-trained-vecs/${dataset}/${lang}-en.${lang}.${model}.txt.gz ../data/${dataset}/${lang}_test.txt 0 ${bp_param} --prefix log_test_${dataset}_${lang}_${model}

		# Classify

		echo -n ${lang}-en ${model} " " >> ${dataset}_results.txt
		python balAPinc_classification.py --training log_tune_${dataset}_${lang}_${model}_${bp_param}_0.0 --test log_test_${dataset}_${lang}_${model}_${bp_param}_0.0 | cut -d' ' -f3 >> ${dataset}_results.txt
		
		done
	done
done
