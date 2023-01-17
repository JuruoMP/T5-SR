#!/bin/bash
date

python -c "import nltk; nltk.download('punkt'); nltk.download('omw-1.4'); nltk.download('wordnet')"
python -m spacy download en
python -m spacy download en_core_web_lg

if [ ! -f logdir/spider_rerank_test/predictions_eval_None.json ]
then
 python seq2seq/run_seq2seq.py  --config_file seq2seq/configs/rerank_test.json
fi

python rerank/schema_data.py
python rerank/ssql2sql.py --type test
python rerank/get_data.py --type test
python rerank/run_rerank.py --type test