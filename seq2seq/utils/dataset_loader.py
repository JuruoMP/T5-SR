import json
import os.path
from typing import Callable, Tuple
import logging
import datasets.load
from datasets.dataset_dict import DatasetDict
from datasets.metric import Metric
from datasets.arrow_dataset import Dataset, concatenate_datasets
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.training_args import TrainingArguments
from seq2seq.utils.args import ModelArguments
from seq2seq.utils.dataset import (
    DataArguments,
    DataTrainingArguments,
    DatasetSplits,
    TrainSplit,
    _prepare_train_split,
    prepare_splits,
    prepare_splits_spider_synonym
)
from seq2seq.utils.spider import spider_add_serialized_schema, spider_pre_process_function
from seq2seq.utils.cosql import cosql_add_serialized_schema, cosql_pre_process_function
from seq2seq.utils.input_collator import ConstrainedInputCollator

logger = logging.getLogger(__name__)


def _log_duplicate_count(dataset: Dataset, dataset_name: str, split: str) -> None:
    d = dataset.to_dict()
    d_t = [tuple((k, tuple(v)) for k, v in zip(d.keys(), vs)) for vs in zip(*d.values())]
    d_t_ = set(d_t)
    num_examples = len(d_t)
    duplicate_count = num_examples - len(d_t_)
    if duplicate_count > 0:
        logger.warning(
            f"The split ``{split}`` of the dataset ``{dataset_name}`` contains {duplicate_count} duplicates out of {num_examples} examples"
        )


def load_dataset(
    data_args: DataArguments,
    model_args: ModelArguments,
    data_training_args: DataTrainingArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizerFast,
) -> Tuple[Metric, DatasetSplits]:
    input_collator = ConstrainedInputCollator(tokenizer)
    #print("dataset_paths:",data_args.dataset_paths["spider"]) #seq2seq/datasets/spider
    print("data_cache_dir:", data_args.data_cache_dir)
    _spider_dataset_dict: Callable[[], DatasetDict] = lambda: datasets.load.load_dataset(
        path=data_args.dataset_paths["spider"], cache_dir=data_args.data_cache_dir
    )
    _spider_other_dataset_dict: Callable[[], DatasetDict] = lambda: datasets.load.load_dataset(
        path=data_args.dataset_paths["spider_other"], cache_dir=data_args.data_cache_dir
    )
    _natsql_dataset_dict: Callable[[], DatasetDict] = lambda: datasets.load.load_dataset(
        path=data_args.dataset_paths["natsql"], cache_dir=data_args.data_cache_dir
    )
    _spider_metric: Callable[[], Metric] = lambda: datasets.load.load_metric(
        path=data_args.metric_paths["spider"], config_name=data_args.metric_config, test_suite_db_dir=data_args.test_suite_db_dir,
        cache_dir=data_args.data_cache_dir
    )
    _spider_add_serialized_schema = lambda ex: spider_add_serialized_schema(
        ex=ex,
        data_training_args=data_training_args,
    )
    _spider_pre_process_function_train = lambda batch, max_source_length, max_target_length: spider_pre_process_function(
        batch=batch,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        data_training_args=data_training_args,
        tokenizer=tokenizer,
        use_synonym= True
    )#train
    _spider_pre_process_function_eval = lambda batch, max_source_length, max_target_length: spider_pre_process_function(
        batch=batch,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        data_training_args=data_training_args,
        tokenizer=tokenizer,
        use_synonym=False
    )#eval
    _spider_pre_process_function = lambda batch, max_source_length, max_target_length: spider_pre_process_function(
        batch=batch,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        data_training_args=data_training_args,
        tokenizer=tokenizer,
        use_synonym=False
    )  # eval
    _dirty_sql_dataset_dict: Callable[[], DatasetDict] = lambda: datasets.load.load_dataset(
        path=data_args.dataset_paths["dirty_sql"], cache_dir=data_args.data_cache_dir
    )
    _cosql_dataset_dict: Callable[[], DatasetDict] = lambda: datasets.load.load_dataset(
        path=data_args.dataset_paths["cosql"], cache_dir=data_args.data_cache_dir
    )
    _cosql_metric: Callable[[], Metric] = lambda: datasets.load.load_metric(
        path=data_args.metric_paths["cosql"], config_name=data_args.metric_config, test_suite_db_dir=data_args.test_suite_db_dir,
        cache_dir=data_args.data_cache_dir
    )
    _dirty_sql_metric: Callable[[], Metric] = lambda: datasets.load.load_metric(
        path=data_args.metric_paths["translate"], config_name=data_args.metric_config, test_suite_db_dir=data_args.test_suite_db_dir
    )
    _cosql_add_serialized_schema = lambda ex: cosql_add_serialized_schema(
        ex=ex,
        data_training_args=data_training_args,
    )
    _cosql_pre_process_function = lambda batch, max_source_length, max_target_length: cosql_pre_process_function(
        batch=batch,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        data_training_args=data_training_args,
        tokenizer=tokenizer,
    )

    _sparc_dataset_dict: Callable[[], DatasetDict] = lambda: datasets.load.load_dataset(
        path=data_args.dataset_paths["sparc"], cache_dir=data_args.data_cache_dir
    )

    _prepare_splits_kwargs = {
        "data_args": data_args,
        "training_args": training_args,
        "data_training_args": data_training_args,
    }

    if data_args.dataset == "spider":
        metric = _spider_metric()
        if data_training_args.use_synonym:
            dataset_splits = prepare_splits_spider_synonym(
                dataset_dict=_spider_dataset_dict(),
                add_serialized_schema=_spider_add_serialized_schema,
                pre_process_function_train=_spider_pre_process_function_train,
                pre_process_function_eval=_spider_pre_process_function_eval,
                **_prepare_splits_kwargs,
            )
        else:
            dataset_splits = prepare_splits(
                dataset_dict=_spider_dataset_dict(),
                add_serialized_schema=_spider_add_serialized_schema,
                pre_process_function=_spider_pre_process_function,
                **_prepare_splits_kwargs,
            )
    elif data_args.dataset == "spider+other":
        metric = _spider_metric()
        if data_training_args.use_synonym:
            spider_dataset_splits = prepare_splits_spider_synonym(
                dataset_dict=_spider_dataset_dict(),
                add_serialized_schema=_spider_add_serialized_schema,
                pre_process_function_train=_spider_pre_process_function_train,
                pre_process_function_eval=_spider_pre_process_function_eval,
                **_prepare_splits_kwargs,
            )
            other_dataset_splits = (
                _prepare_train_split(
                    dataset=_spider_other_dataset_dict()["train"],
                    data_training_args=data_training_args,
                    add_serialized_schema=_spider_add_serialized_schema,
                    pre_process_function=_spider_pre_process_function_train,
                )
            )
        else:
            spider_dataset_splits = prepare_splits(
                dataset_dict=_spider_dataset_dict(),
                add_serialized_schema=_spider_add_serialized_schema,
                pre_process_function=_spider_pre_process_function,
                **_prepare_splits_kwargs,
            )
            other_dataset_splits = (
                _prepare_train_split(
                    dataset=_spider_other_dataset_dict()["train"],
                    data_training_args=data_training_args,
                    add_serialized_schema=_spider_add_serialized_schema,
                    pre_process_function=_spider_pre_process_function,
                )
            )
        dataset: Dataset = concatenate_datasets(
            dsets=[spider_dataset_splits.train_split.dataset, other_dataset_splits.dataset]
        )
        train_split = TrainSplit(
            dataset=dataset,
            schemas=spider_dataset_splits.train_split.schemas,
        )
        dataset_splits = DatasetSplits(
            train_split=train_split,
            eval_split=spider_dataset_splits.eval_split,
            test_splits=spider_dataset_splits.test_splits,
            schemas=spider_dataset_splits.train_split.schemas,
        )
    elif data_args.dataset == "natsql":
        metric = _spider_metric()
        dataset_splits = prepare_splits(
            dataset_dict=_natsql_dataset_dict(),
            add_serialized_schema=_spider_add_serialized_schema,
            pre_process_function=_spider_pre_process_function,
            **_prepare_splits_kwargs,
        )
    elif data_args.dataset == "sparc":
        metric = _cosql_metric()
        dataset_splits = prepare_splits(
            dataset_dict=_sparc_dataset_dict(),
            add_serialized_schema=_cosql_add_serialized_schema,
            pre_process_function=_cosql_pre_process_function,
            **_prepare_splits_kwargs,
        )
    elif data_args.dataset == "cosql":
        metric = _cosql_metric()
        dataset_splits = prepare_splits(
            dataset_dict=_cosql_dataset_dict(),
            add_serialized_schema=_cosql_add_serialized_schema,
            pre_process_function=_cosql_pre_process_function,
            **_prepare_splits_kwargs,
        )
    elif data_args.dataset == "dirty_sql":  # 合成数据集 hanchu
        metric = _dirty_sql_metric()
        dataset_splits = prepare_splits(
            dataset_dict=_dirty_sql_dataset_dict(),
            add_serialized_schema=_spider_add_serialized_schema,
            pre_process_function=_spider_pre_process_function,
            **_prepare_splits_kwargs,
        )
    elif data_args.dataset == "cosql+spider":
        metric = _cosql_metric()
        cosql_dataset_splits = prepare_splits(
            dataset_dict=_cosql_dataset_dict(),
            add_serialized_schema=_cosql_add_serialized_schema,
            pre_process_function=_cosql_pre_process_function,
            **_prepare_splits_kwargs,
        )
        spider_training_split = (
            _prepare_train_split(
                dataset=_spider_dataset_dict()["train"],
                data_training_args=data_training_args,
                add_serialized_schema=_spider_add_serialized_schema,
                pre_process_function=_spider_pre_process_function,
            )
            if training_args.do_train
            else None
        )
        if cosql_dataset_splits.train_split is None and spider_training_split is None:
            train_split = None
        elif cosql_dataset_splits.train_split is None:
            train_split = spider_training_split
        elif spider_training_split is None:
            train_split = cosql_dataset_splits.train_split
        else:
            dataset: Dataset = concatenate_datasets(
                dsets=[cosql_dataset_splits.train_split.dataset, spider_training_split.dataset]
            )
            train_split = TrainSplit(
                dataset=dataset,
                schemas={**spider_training_split.schemas, **cosql_dataset_splits.train_split.schemas},
            )
        schemas = {
            **cosql_dataset_splits.schemas,
            **(spider_training_split.schemas if spider_training_split is not None else {}),
        }
        dataset_splits = DatasetSplits(
            train_split=train_split,
            eval_split=cosql_dataset_splits.eval_split,
            test_splits=cosql_dataset_splits.test_splits,
            schemas=schemas,
        )
    else:
        raise NotImplementedError()

    if dataset_splits.train_split is not None:
        _log_duplicate_count(dataset=dataset_splits.train_split.dataset, dataset_name=data_args.dataset, split="train")
    if dataset_splits.eval_split is not None:
        _log_duplicate_count(dataset=dataset_splits.eval_split.dataset, dataset_name=data_args.dataset, split="eval")
    if dataset_splits.test_splits is not None:
        for section, split in dataset_splits.test_splits.items():
            _log_duplicate_count(dataset=split.dataset, dataset_name=data_args.dataset, split=section)

    return metric, dataset_splits