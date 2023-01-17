import collections
import json
import os
from typing import Dict, List, Optional, NamedTuple
import transformers.trainer_seq2seq
from transformers.trainer_utils import PredictionOutput, speed_metrics
from datasets.arrow_dataset import Dataset
from datasets.metric import Metric
import numpy as np
import time
import torch
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
import traceback

from transformers.utils import logging
from transformers.trainer_utils import EvalLoopOutput, denumpify_detensorize
from transformers.trainer_pt_utils import (find_batch_size, nested_concat, nested_numpify, nested_truncate, IterableDatasetShard)
import sys
sys.path.insert(0, './')
sys.path.insert(0, '../')
this_dir = os.path.dirname(__file__)
print("this_dir:",this_dir)
from seq2seq.lf_util.ssql.ssql_parser import SqlParser
from seq2seq.lf_util.ssql.ssql_fix_parser import SqlFixParser

from packaging import version
if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True

logger = logging.get_logger(__name__)


class EvalPrediction(NamedTuple):
    predictions: List[str]
    label_ids: np.ndarray
    metas: List[dict]

# todo: apply constrained decoding for constrained lm
class ConstrainedSeq2SeqTrainer(transformers.trainer_seq2seq.Seq2SeqTrainer):
    def __init__(
        self,
        metric: Metric,
        *args,
        eval_examples: Optional[Dataset] = None,
        ignore_pad_token_for_loss: bool = True,
        target_with_db_id: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.metric = metric
        self.eval_examples = eval_examples
        self.compute_metrics = self._compute_metrics
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.target_with_db_id = target_with_db_id

    def _compute_metrics(self, eval_prediction: EvalPrediction) -> dict:
        raise NotImplementedError()

    def _post_process_function(
        self, examples: Dataset, features: Dataset, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:
        raise NotImplementedError()

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        eval_examples: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        max_time: Optional[int] = None,
        num_beams: Optional[int] = None,
        num_return_sequences: Optional[int] = 1,#第一个 !!!!!!魔法数字!!!!!!
        output_scores: Optional[bool] = True,
        return_dict_in_generate: Optional[bool] = True
    ) -> Dict[str, float]:
        self._max_length = max_length
        self._max_time = max_time
        self._num_beams = num_beams

        # hanchu print beam
        self.num_return_sequences = num_return_sequences
        self.output_scores = output_scores
        self.return_dict_in_generate = return_dict_in_generate

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        start_time = time.time()

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output: PredictionOutput = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
                num_return_sequences = num_return_sequences,
                output_scores = output_scores,
                return_dict_in_generate = return_dict_in_generate
            )
        finally:
            self.compute_metrics = compute_metrics

        # We might have removed columns from the dataset so we put them back.
        if isinstance(eval_dataset, Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format["type"],
                columns=list(eval_dataset.features.keys()),
            )

        if eval_examples is not None and eval_dataset is not None and self.compute_metrics is not None:
            eval_preds = self._post_process_function(
                eval_examples,
                eval_dataset,
                output.predictions,
                "eval_{}".format(self.state.epoch),
                output.scores
            )
            output.metrics.update(self.compute_metrics(eval_preds))

        n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics


class ConstrainedSpiderSeqTrainer(ConstrainedSeq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        self.train_to_generate_sql = kwargs['train_to_generate_sql']
        self.is_sql_fix_parser = kwargs['is_sql_fix_parser']
        print("train_to_generate_sql:",self.train_to_generate_sql)
        print("is_sql_fix_parser:",self.is_sql_fix_parser)
        del kwargs['train_to_generate_sql']
        del kwargs['is_sql_fix_parser']
        super().__init__(*args, **kwargs)
        parms = kwargs.get('args')
        self.sql_parser = SqlParser(table_path='data/spider/tables.json', db_dir='data/database')
        if parms.do_train:
            self.sql_fix_parser = self.sql_parser
        else:
            self.sql_fix_parser = SqlFixParser(self.sql_parser)


    def _sql_float2int(self,sql: str):
        tokens = sql.split(' ')
        new_tokens = []
        for t in tokens:
            try:
                f_t = float(t)
                new_tokens.append(str(int(f_t)))
            except:
                new_tokens.append(t)
        return ' '.join(new_tokens)

    def _post_process_function(
            self, examples: Dataset, features: Dataset, predictions: np.ndarray, stage: str, scores: np.ndarray
    ) -> EvalPrediction:
        inputs = self.tokenizer.batch_decode([f["input_ids"] for f in features], skip_special_tokens=True)
        label_ids = [f["labels"] for f in features]
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            _label_ids = np.where(label_ids != -100, label_ids, self.tokenizer.pad_token_id)
        decoded_label_ids = self.tokenizer.batch_decode(_label_ids, skip_special_tokens=True)
        metas = []
        for x, context, label in zip(examples, inputs, decoded_label_ids):
            # for x, context, label, score in zip(examples, inputs, decoded_label_ids, scores):
            #print(x)
            for i in range(self.num_return_sequences):
                metas.append({
                    "raw_query": x["raw_query"] if "raw_query" in x else "",
                    "query": x["query"] if "query" in x else "",
                    "question": x["question"] if "question" in x else "",
                    "context": context,
                    "label": label,
                    # "score":score,
                    "db_id": x["db_id"] if "db_id" in x else "",
                    "db_path": x["db_path"] if "db_path" in x else "",
                    "db_table_names": x["db_table_names"] if "db_table_names" in x else "",
                    "db_column_names": x["db_column_names"] if "db_column_names" in x else "",
                    "db_foreign_keys": x["db_foreign_keys"] if "db_foreign_keys" in x else "",
                    "beam_i": i,
                    "num_return_sequences": self.num_return_sequences
                })
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # print(len(metas))
        # print(len(predictions))
        assert len(metas) == len(predictions)
        # print("writing.....")
        # print(stage)

        with open(f"{self.args.output_dir}/predictions_{stage}.json", "w") as f:
            json.dump(
                [dict(**{"prediction": prediction}, **{"score": str(score)}, **meta) for prediction, meta, score in
                 zip(predictions, metas, scores)],
                f,
                indent=4,
            )
        return EvalPrediction(predictions=predictions, label_ids=label_ids, metas=metas)

    def _compute_metrics(self, eval_prediction: EvalPrediction) -> dict:
        predictions, label_ids, metas = eval_prediction
        cnt_dict = collections.defaultdict(int)
        all_pred_list, all_status_list, all_exec_match_list = [], [], []
        """
        以下是一个巨大的if, 阅读者请折叠
        """
        if self.train_to_generate_sql == 'mid_sql':  # 中间语言
            #print("predictions:", str(len(predictions)))
            for i in range(len(predictions)):
                cnt_dict['all'] = cnt_dict['all'] + 1
                pred = predictions[i]
                status = None
                try:
                    _, pred_sql = pred.split('|')
                except:
                    pred_sql = 'ERROR'
                    status = 'no-splitter'

                db_name = metas[i]['db_id']
                try:
                    pred_sql_dict = self.sql_parser.sql_to_dict(db_name, pred_sql)
                    pred_raw_sql = self.sql_parser.dict_to_raw_sql(db_name, pred_sql_dict)
                except:
                    cnt_dict['illegal'] = cnt_dict['illegal'] + 1
                    try:
                        pred_sql_dict = self.sql_fix_parser.sql_to_dict(db_name, pred_sql)
                        pred_raw_sql = self.sql_fix_parser.dict_to_raw_sql(db_name, pred_sql_dict)
                        status = 'fix'
                    except Exception as e:
                        pred_sql_dict = {}
                        pred_raw_sql = ''
                        if not status:
                            status = 'illegal'

                try:
                    #print(pred_raw_sql)
                    if self.sql_parser.check_equal_script(db_name, pred_raw_sql, metas[i]['raw_query']):
                        cnt_dict['correct'] = cnt_dict['correct'] + 1
                        if not status:
                            status = 'correct'
                        else:
                            status += '_correct'
                            cnt_dict[status] = cnt_dict[status] + 1
                    else:
                        cnt_dict['wrong'] = cnt_dict['wrong'] + 1
                        if not status:
                            status = 'wrong'
                        else:
                            status += '_wrong'
                            cnt_dict[status] = cnt_dict[status] + 1
                except Exception as e:
                    #traceback.print_exc()
                    if not status:
                        status = 'unknown'
                all_pred_list.append((db_name, pred_sql.strip(), metas[i]['raw_query']))
                all_status_list.append(status)

                try:
                    pred_raw_sql = self._sql_float2int(pred_raw_sql)
                    exec_match = self.sql_parser.check_exec_match_script(
                        db_name, pred_raw_sql, metas[i]['raw_query'], pred_sql_dict,
                        self.sql_parser.raw_sql_to_dict(db_name, metas[i]['raw_query']))
                except:
                    exec_match = False
                all_exec_match_list.append(exec_match)

            os.makedirs('logdir/spider_log', exist_ok=True)
            with open('logdir/spider_log/pred.txt', 'w') as fw:
                for (db_id, pred, gold), status in zip(all_pred_list, all_status_list):
                    fw.write(f"{db_id}\t{status}\t{pred}\t{gold}\n")
            eval_exact_match = cnt_dict['correct'] / cnt_dict['all']
            print(f'Exact match = {cnt_dict["correct"]} / {cnt_dict["all"]} with {cnt_dict["illegal"]} illegal cases')
            print(f'cnt_dict = {cnt_dict}')
            eval_exec_match = 0.0
            if all_exec_match_list:
                eval_exec_match = len([x for x in all_exec_match_list if x is True]) / len(all_exec_match_list)
            if self.train_to_generate_sql == 'mid_sql':
                if self.is_sql_fix_parser and eval_exact_match > 0.5:
                    self.sql_fix_parser = SqlFixParser(self.sql_parser)
            return {'eval_exact_match': eval_exact_match, 'eval_exec_match': eval_exec_match}

        elif self.train_to_generate_sql == 'sql':  # raw sql
            predictions, label_ids, metas = eval_prediction
            all_pred_list, all_status_list, all_exec_match_list = [], [], []
            for i in range(len(predictions)):
                cnt_dict['all'] += 1
                pred = predictions[i]
                status = None
                try:
                    _, pred_sql = pred.split('|')
                except:
                    pred_sql = 'ERROR'
                    status = 'no-splitter'

                db_name = metas[i]['db_id']

                try:
                    # if self.sql_parser.check_equal(pred_sql_dict, gold_sql_dict):
                    if self.sql_parser.check_equal_script(db_name, pred_sql, metas[i]['raw_query']):
                        cnt_dict['correct'] += 1
                        if not status:
                            status = 'correct'
                    else:
                        cnt_dict['wrong'] += 1
                        if not status:
                            status = 'wrong'
                except:
                    if not status:
                        status = 'illegal'
                all_pred_list.append((db_name, pred_sql.strip(), metas[i]['query']))
                all_status_list.append(status)

                try:
                    pred_sql_dict = self.sql_parser.raw_sql_to_dict(db_name, pred_sql)
                    pred_sql = self._sql_float2int(pred_sql)
                    exec_match = self.sql_parser.check_exec_match_script(
                        db_name, pred_sql, metas[i]['raw_query'], pred_sql_dict,
                        self.sql_parser.raw_sql_to_dict(db_name, metas[i]['raw_query']))
                except:
                    exec_match = False
                all_exec_match_list.append(exec_match)

            os.makedirs('logdir/spider_sql_raw_log', exist_ok=True)
            with open('logdir/spider_sql_raw_log/pred.txt', 'w') as fw:
                for (db_id, pred, gold), status in zip(all_pred_list, all_status_list):
                    fw.write(f"{db_id}\t{status}\t{pred}\t{gold}\n")
            eval_exact_match = cnt_dict['correct'] / cnt_dict['all']
            print(f'Exact match = {cnt_dict["correct"]} / {cnt_dict["all"]} with {cnt_dict["illegal"]} illegal cases')
            print(f'cnt_dict = {cnt_dict}')
            eval_exec_match = 0.0
            if all_exec_match_list:
                eval_exec_match = len([x for x in all_exec_match_list if x is True]) / len(all_exec_match_list)

            if eval_exact_match > 0.5:
                self.sql_fix_parser = SqlFixParser(self.sql_parser)

            return {'eval_exact_match': eval_exact_match, 'eval_exec_match': eval_exec_match}
        elif self.train_to_generate_sql == 'dirty_sql':
            predictions, label_ids, metas = eval_prediction
            scores = []
            all_pred_list, all_status_list, all_exec_match_list = [], [], []
            for i in range(len(predictions)):
                pred = predictions[i]
                status = None
                try:
                    _, pred_sql = pred.split('|')
                except:
                    pred_sql = 'ERROR'
                    status = 'no-splitter'

                db_name = metas[i]['db_id']
                final_pred_sql = pred_sql

                tokenized_gold = word_tokenize(metas[i]['query'].strip().lower())
                tokenized_predict = word_tokenize(final_pred_sql.strip().lower())
                score = sentence_bleu([tokenized_gold], tokenized_predict)
                # if score < 1:
                #     print("=======================")
                #     print(metas[i]['query'].strip().lower())
                #     print(final_pred_sql.strip().lower())
                #     print(score)

                scores.append(score)
            scores_np = np.array(scores, np.float32)
            eval_exact_match = np.sum(scores_np >= 0.95)/len(scores)
            return {'eval_exact_match': eval_exact_match, 'eval_exec_match': 0.0}



class ConstrainedNatsqlSeqTrainer(ConstrainedSeq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from seq2seq.lf_util.natsql.natsql_parser import NatsqlParser
        from seq2seq.lf_util.natsql.natsql_fix_parser import NatsqlFixParser
        self.sql_parser = NatsqlFixParser('data/natsql/tables_for_natsql.json', 'data/database')

    def _post_process_function(
        self, examples: Dataset, features: Dataset, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:
        inputs = self.tokenizer.batch_decode([f["input_ids"] for f in features], skip_special_tokens=True)
        label_ids = [f["labels"] for f in features]
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            _label_ids = np.where(label_ids != -100, label_ids, self.tokenizer.pad_token_id)
        decoded_label_ids = self.tokenizer.batch_decode(_label_ids, skip_special_tokens=True)
        metas = [
            {
                "raw_query": x["raw_query"],
                "query": x["query"],
                "question": x["question"],
                "context": context,
                "label": label,
                "db_id": x["db_id"],
                "db_path": x["db_path"],
                "db_table_names": x["db_table_names"],
                "db_column_names": x["db_column_names"],
                "db_foreign_keys": x["db_foreign_keys"],
            }
            for x, context, label in zip(examples, inputs, decoded_label_ids)
        ]
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        assert len(metas) == len(predictions)
        with open(f"{self.args.output_dir}/predictions_{stage}.json", "w") as f:
            json.dump(
                [dict(**{"prediction": prediction}, **meta) for prediction, meta in zip(predictions, metas)],
                f,
                indent=4,
            )
        return EvalPrediction(predictions=predictions, label_ids=label_ids, metas=metas)

    def _compute_metrics(self, eval_prediction: EvalPrediction) -> dict:
        predictions, label_ids, metas = eval_prediction
        cnt_dict = {}
        all_pred_list, all_status_list, all_exec_match_list = [], [], []
        for i in range(len(predictions)):
            cnt_dict['all'] = cnt_dict.get('all', 0) + 1
            pred = predictions[i]
            status = None
            try:
                _, pred_sql = pred.split('|')
            except:
                pred_sql = 'ERROR'
                status = 'no-splitter'

            db_name = metas[i]['db_id']
            try:
                pred_raw_sql = self.sql_parser.natsql_to_sql(db_name, pred_sql)
            except:
                pred_raw_sql = ''
                status = 'illegal'
                cnt_dict['illegal'] = cnt_dict.get('illegal', 0) + 1

            if not status:
                try:
                    if self.sql_parser.check_equal_script(db_name, pred_raw_sql, metas[i]['raw_query']):
                        cnt_dict['correct'] = cnt_dict.get('correct', 0) + 1
                        status = 'correct'
                    else:
                        cnt_dict['wrong'] = cnt_dict.get('wrong', 0) + 1
                        status = 'wrong'
                except:
                    if not status:
                        status = 'unknown'
            all_pred_list.append((db_name, pred_sql.strip(), metas[i]['raw_query']))
            all_status_list.append(status)

            try:
                pred_sql_dict = self.sql_parser.raw_sql_to_dict(db_name, pred_raw_sql)
                exec_match = self.sql_parser.check_exec_match_script(
                    db_name, pred_raw_sql, metas[i]['raw_query'], pred_sql_dict,
                    self.sql_parser.raw_sql_to_dict(db_name, metas[i]['raw_query']))
            except:
                exec_match = False
            all_exec_match_list.append(exec_match)

        os.makedirs('logdir/natsql_log', exist_ok=True)
        with open('logdir/natsql_log/pred.txt', 'w') as fw:
            for (db_id, pred, gold), qm_status, em_status in zip(all_pred_list, all_status_list, all_exec_match_list):
                em_status = 'correct' if em_status is True else 'wrong'
                fw.write(f"{db_id}\tQM={qm_status}\tEM={em_status}\t{pred}\t{gold}\n")
        eval_exact_match = cnt_dict['correct'] / cnt_dict['all']
        print(f'Exact match = {cnt_dict["correct"]} / {cnt_dict["all"]} with {cnt_dict["illegal"]} illegal cases')
        print(f'cnt_dict = {cnt_dict}')
        eval_exec_match = -1
        if all_exec_match_list:
            eval_exec_match = len([x for x in all_exec_match_list if x is True]) / len(all_exec_match_list)

        # if eval_exact_match > 0.5:
        #     self.sql_fix_parser = SqlFixParser(self.sql_parser)

        return {'eval_exact_match': eval_exact_match, 'eval_exec_match': eval_exec_match}


class ConstrainedCoSQLSeqTrainer(ConstrainedSeq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        from seq2seq.lf_util.ssql.ssql_parser import SqlParser
        super().__init__(*args, **kwargs)
        self.sql_parser = SqlParser('data/cosql/tables.json', 'data/database')

    def _post_process_function(
        self, examples: Dataset, features: Dataset, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:
        inputs = self.tokenizer.batch_decode([f["input_ids"] for f in features], skip_special_tokens=True)
        label_ids = [f["labels"] for f in features]
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            _label_ids = np.where(label_ids != -100, label_ids, self.tokenizer.pad_token_id)
        decoded_label_ids = self.tokenizer.batch_decode(_label_ids, skip_special_tokens=True)
        metas = [
            {
                "diag_id": x["diag_id"],
                "raw_query": x["raw_query"],
                "query": x["query"],
                "utterances": x["utterances"],
                "turn_idx": x["turn_idx"],
                "context": context,
                "label": label,
                "db_id": x["db_id"],
                "db_path": x["db_path"],
                "db_table_names": x["db_table_names"],
                "db_column_names": x["db_column_names"],
                "db_foreign_keys": x["db_foreign_keys"],
            }
            for x, context, label in zip(examples, inputs, decoded_label_ids)
        ]
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        assert len(metas) == len(predictions)
        with open(f"{self.args.output_dir}/predictions_{stage}.json", "w") as f:
            json.dump(
                [dict(**{"prediction": prediction}, **meta) for prediction, meta in zip(predictions, metas)],
                f,
                indent=4,
            )
        return EvalPrediction(predictions=predictions, label_ids=label_ids, metas=metas)

    def _compute_metrics(self, eval_prediction: EvalPrediction) -> dict:
        predictions, label_ids, metas = eval_prediction
        cnt_all, cnt_correct = 0, 0
        cnt_pred_illegal, cnt_gold_illegal = 0, 0
        all_pred_list = []
        all_status_list = []
        error_list = []
        diag_correct_map = {}
        for i in range(len(predictions)):
            pred = predictions[i]
            status = None
            try:
                _, pred_sql = pred.split('|')
            except:
                pred_sql = 'ERROR'
                status = 'no-splitter'
            db_name = metas[i]['db_id']
            all_pred_list.append((db_name, pred_sql.strip(), metas[i]['query']))
            try:
                pred_sql_dict = self.sql_parser.sql_to_dict(db_name, pred_sql)
                pred_raw_sql = self.sql_parser.dict_to_raw_sql(db_name, pred_sql_dict)
            except:
                cnt_pred_illegal += 1
                pred_raw_sql = ''
                if not status:
                    status = 'illegal'

            cnt_all += 1
            try:
                # if self.sql_parser.check_equal(pred_sql_dict, gold_sql_dict):
                if self.sql_parser.check_equal_script(db_name, pred_raw_sql, metas[i]['raw_query']):
                    cnt_correct += 1
                    if not status:
                        status = 'correct'
                else:
                    error_list.append((db_name, pred_raw_sql, metas[i]['raw_query']))
                    if not status:
                        status = 'wrong'
            except:
                if not status:
                    status = 'unknown'
            all_status_list.append(status)
            if metas[i]['diag_id'] != -1:
                diag_label = diag_correct_map.get(metas[i]['diag_id'], True)
                if status != 'correct':
                    diag_label = False
                diag_correct_map[metas[i]['diag_id']] = diag_label
        os.makedirs('logdir/sparc_log', exist_ok=True)
        with open('logdir/sparc_log/pred.txt', 'w') as fw:
            for (db_id, pred, gold), status in zip(all_pred_list, all_status_list):
                fw.write(f"{db_id}\t{status}\t{pred}\t{gold}\n")
        eval_exact_match = cnt_correct / cnt_all
        n_diags, n_correct_diags = len(diag_correct_map), len([x for x in diag_correct_map.values() if x is True])
        interaction_match = n_correct_diags / n_diags
        print(f'QM = {eval_exact_match:.4f} with {cnt_pred_illegal} / {cnt_gold_illegal} illegal cases, '
              f'IM = {interaction_match:.4f}')
        if eval_exact_match > 0.35:
            self.sql_parser = SqlFixParser(self.sql_parser)
        return {'eval_exact_match': eval_exact_match, 'eval_im': interaction_match}
