from typing import Dict, Any
# from third_party.spider import evaluation as spider_evaluation
from nltk import word_tokenize
from datasets import load_metric
from nltk.translate.bleu_score import sentence_bleu
import datasets
from typing import Optional, Union

_DESCRIPTION = """
translate metrics.
"""

_KWARGS_DESCRIPTION = """
"""

_CITATION = """ see nltk.translate.bleu_score
"""

_URL = "https://drive.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0"


class TranslateEvaluator(datasets.Metric):
    def __init__(self,
                 config_name: Optional[str] = None,
                 keep_in_memory: bool = False,
                 cache_dir: Optional[str] = None,
                 num_process: int = 1,
                 process_id: int = 0,
                 seed: Optional[int] = None,
                 experiment_id: Optional[str] = None,
                 max_concurrent_cache_files: int = 10000,
                 timeout: Union[int, float] = 100,
                 **kwargs
                 ):
        # self.metrics = load_metric('metrics/sacrebleu/sacrebleu.py')
        # https://huggingface.co/docs/datasets/using_metrics.html#adding-predictions-and-references
        print("init TranslateEvaluator")
        super().__init__(
            config_name=config_name,
            keep_in_memory=keep_in_memory,
            cache_dir=cache_dir,
            num_process=num_process,
            process_id=process_id,
            seed=seed,
            experiment_id=experiment_id,
            max_concurrent_cache_files=max_concurrent_cache_files,
            timeout=timeout,
            **kwargs
        )

    def _info(self):
        if self.config_name not in [
            "exact_match",
            "test_suite",
            "both",
        ]:
            raise KeyError(
                "You should supply a configuration name selected in " '["exact_match", "test_suite", "both"]'
            )
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": {
                        "query": datasets.Value("string"),
                        "question": datasets.Value("string"),
                        "context": datasets.Value("string"),
                        "label": datasets.Value("string"),
                        "db_id": datasets.Value("string"),
                        "db_path": datasets.Value("string"),
                        "db_table_names": datasets.features.Sequence(datasets.Value("string")),
                        "db_column_names": datasets.features.Sequence(
                            {
                                "table_id": datasets.Value("int32"),
                                "column_name": datasets.Value("string"),
                            }
                        ),
                        "db_foreign_keys": datasets.features.Sequence(
                            {
                                "column_id": datasets.Value("int32"),
                                "other_column_id": datasets.Value("int32"),
                            }
                        ),
                    },
                }
            ),
            reference_urls=[_URL],
        )

    def _compute(self, predictions, references) -> Dict[str, Any]:
        # reference_batch = [[r['skeleton']] for r in references]

        scores = []

        # sys_batch = []
        # reference_batch = []
        for prediction, reference in zip(predictions, references):
            gold = reference["skeleton"]

            tokenized_gold = self.tokenize(gold)
            tokenized_predict = self.tokenize(prediction)
            # reference_batch.append([tokenized_gold])
            # sys_batch.append(tokenized_predict)
            score = sentence_bleu([tokenized_gold], tokenized_predict)
            print("piece:")
            print(tokenized_gold)
            print(tokenized_predict)
            print(score)
            scores.append(score)

        # self.metrics.add_batch(predictions=sys_batch, references=reference_batch)
        # scores = self.metrics.compute()
        return {
            "exact_match": scores
        }

    def evaluate_one(self, gold, prediction):
        # self.tokenize(gold)
        # self.tokenize(gold)
        return

    def tokenize(self, string):
        string = str(string)
        # string = string.replace(
        #     "'", '"'
        # )
        # ensures all string values wrapped by "" problem??
        # quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
        # if len(quote_idxs) % 2 == 0:
        #     print("Unexpected quote:",string)

        # keep string value as token
        # vals = {}
        # for i in range(len(quote_idxs) - 1, -1, -2):
        #     qidx1 = quote_idxs[i - 1]
        #     qidx2 = quote_idxs[i]
        #     val = string[qidx1 : qidx2 + 1]
        #     key = "__val_{}_{}__".format(qidx1, qidx2)
        #     string = string[:qidx1] + key + string[qidx2 + 1 :]
        #     vals[key] = val

        toks = [word.lower() for word in word_tokenize(string)]
        # replace with string value token
        # for i in range(len(toks)):
        #     if toks[i] in vals:
        #         toks[i] = vals[toks[i]]

        # find if there exists !=, >=, <=, <>`
        eq_idxs = [idx for idx, tok in enumerate(toks) if tok in ("=", ">")]
        eq_idxs.reverse()
        prefix = ("!", ">", "<")
        for eq_idx in eq_idxs:
            pre_tok = toks[eq_idx - 1]
            if pre_tok in prefix:
                toks = toks[: eq_idx - 1] + [pre_tok + toks[eq_idx]] + toks[eq_idx + 1:]

        return toks
