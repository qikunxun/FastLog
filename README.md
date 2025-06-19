## Code of our paper "Reconstructing TensorLog for Scalable End-to-end Rule Learning"

## Prerequisites

* Python 3.9
* pytorch 2.4.0
* Pandas
* Numpy
* pytorch_scatter

### Datasets

We used Family, Kinship, UMLS, WN18RR, FB15k-237, YAGO3-10, FB15k-237 (Inductive), NELL-995 (Inductive), Wikidata5M, Freebase in our experiments.

| Datasets           | Download Links                                                       |
|--------------------|----------------------------------------------------------------------|
| Family             | https://github.com/fanyangxyz/Neural-LP                        |
| Kinship            | https://github.com/DeepGraphLearning/RNNLogic                        |
| UMLS               | https://github.com/DeepGraphLearning/RNNLogic                        |
| WN18RR             | https://github.com/DeepGraphLearning/RNNLogic   |
| FB15k-237          | https://github.com/DeepGraphLearning/RNNLogic   |
| YAGO3-10           | https://huggingface.co/datasets/VLyb/YAGO3-10   |
| FB15k-237 (Inductive)         | https://github.com/kkteru/grail   |
| NELL-995 (Inductive)           | https://github.com/kkteru/grail   |
| Wikidata5M         | https://web.informatik.uni-mannheim.de/pi1/kge-datasets/wikidata5m.tar.gz   |
| Freebase           | http://web.informatik.uni-mannheim.de/pi1/kge-datasets/freebase.tar.gz   |

### Models

We use four models in our experiments.

| Models             | Code Download Links (original)                  |
|--------------------|-------------------------------------------------|
| NeuralLP           | https://github.com/fanyangxyz/Neural-LP         |
| DRUM               | https://github.com/alisadeghian/DRUM            |
| smDRUM             | https://github.com/xiaxia-wang/FaithfulRE       |
| mmDRUM             | https://github.com/xiaxia-wang/FaithfulRE       |

## Use examples (Transductive datasets)

For Family, Kinship, and UMLS, you can run the FastLog-enhanced models as bellow:

```sh
python -u main.py --data_dir ../data/family/ --exps_dir ../logs/exps_family_drum_seed1234/ --exp_name family --batch_size 32 --length 3 --max_epoch 10 --dropout 0. --use_gpu --gpu_id 0  --step 3 --do_train --do_test --max_time -1 --min_time -1 --learning_rate 1e-3 --accum_step 1 --early_stop --raw --model_name DRUM --seed 1234
```

For WN18RR, FB15k-237, and YAGO3-10, you can run the FastLog-enhanced models as bellow:

```sh
python -u main.py --data_dir ../data/wn18rr/ --exps_dir ../logs/exps_wn18rr_drum_seed1234/ --exp_name wn18rr --batch_size 32 --length 3 --max_epoch 10 --dropout 0. --use_gpu --gpu_id 0  --step 3 --do_train --do_test --max_time -1 --min_time -1 --learning_rate 1e-3 --accum_step 1 --early_stop --raw --model_name DRUM --seed 1234 --use_topk
```

For Wikidata5M, you can run the FastLog-enhanced models as bellow:

```sh
python -u main.py --data_dir ../data/wikidata5m/ --exps_dir ../logs/exps_wikidata5m_drum_seed1234/ --exp_name wikidata5m --batch_size 16 --length 3 --max_epoch 10 --dropout 0. --use_gpu --gpu_id 0  --step 3 --do_train --do_test --max_time 20000 --min_time -1 --learning_rate 1e-3 --accum_step 1 --early_stop --raw --model_name DRUM --seed 1234 --use_topk --sparse
```

For Freebase, you can run the FastLog-enhanced models as bellow:

```sh
python -u main.py --data_dir ../data/freebase/ --exps_dir ../logs/exps_freebase_drum_seed1234/ --exp_name freebase --batch_size 1 --length 3 --max_epoch 10 --dropout 0. --use_gpu --gpu_id 0  --step 3 --do_train --do_test --max_time 20000 --min_time -1 --learning_rate 1e-3 --accum_step 1 --early_stop --raw --model_name DRUM --seed 1234 --use_topk --sparse
```

## Use examples (Inductive Setting)

For FB15k-237 (Inductive) and NELL-995 (Inductive), you can train the FastLog-enhanced models as bellow:

```sh
python -u main.py --data_dir ../data/inductive/fb237_v1/ --exps_dir ../logs/exps_fb237_v1_drum_seed1234_l6/ --exp_name fb237_v1 --batch_size 32 --length 3 --max_epoch 10 --dropout 0. --use_gpu --gpu_id 0  --step 6 --do_train --do_test --max_time -1 --min_time -1 --learning_rate 1e-3 --accum_step 1 --early_stop --raw --model_name DRUM --seed 1234
```

For evaluating the trained model on inductive datasets, you can run:

```sh
python -u inducve_eval.py --data_dir ../data/inductive/fb237_v1_ind/ --exps_dir ../logs/exps_fb237_v1_drum_seed1234_l6/ --exp_name fb237_v1 --batch_size 32 --length 3 --max_epoch 10 --dropout 0. --use_gpu --gpu_id 0  --step 6 --do_test --max_time -1 --min_time -1 --learning_rate 1e-3 --accum_step 1 --early_stop --raw --model_name DRUM --seed 1234
```
