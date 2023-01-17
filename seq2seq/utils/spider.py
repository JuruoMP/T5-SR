import json
import numpy as np
from typing import Optional
from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from seq2seq.utils.dataset import DataTrainingArguments, normalize, serialize_schema
from seq2seq.utils.trainer import Seq2SeqTrainer, EvalPrediction
import copy,re

def spider_get_input(
    question: str,
    serialized_schema: str,
    prefix: str,
    use_instruction:bool
) -> str:
    if use_instruction:
        return prefix + "The query is " + question.strip() + " the content of database : " + serialized_schema.strip()
    else:
        return prefix + question.strip() + " " + serialized_schema.strip()


def spider_get_target(
    query: str,
    db_id: str,
    normalize_query: bool,
    target_with_db_id: bool,
) -> str:
    _normalize = normalize if normalize_query else (lambda x: x)
    return f"{db_id} | {_normalize(query)}" if target_with_db_id else _normalize(query)


def spider_add_serialized_schema(ex: dict, data_training_args: DataTrainingArguments) -> dict:
    serialized_schema = serialize_schema(
        question=ex["question"],
        db_path=ex["db_path"],
        db_id=ex["db_id"],
        db_column_names=ex["db_column_names"],
        db_table_names=ex["db_table_names"],
        schema_serialization_type=data_training_args.schema_serialization_type,
        schema_serialization_randomized=data_training_args.schema_serialization_randomized,
        schema_serialization_with_db_id=data_training_args.schema_serialization_with_db_id,
        schema_serialization_with_db_content=data_training_args.schema_serialization_with_db_content,
        normalize_query=data_training_args.normalize_query,
        use_instruction=data_training_args.use_instruction,
    )
    return {"serialized_schema": serialized_schema}

def add_synonym(synonym_dict_path,inputs,batch):
    with open(synonym_dict_path, "r") as f:
        all_dict = json.load(f)
    synonym_dict = all_dict["synonym_dict"]
    columns_name_dict = all_dict["synonym_dict_for_sql_col"]
    tables_name_dict = all_dict["tables_name_dict"]
    synonym_inputs, query_synonym = [], []
    for index, i in enumerate(inputs):
        question_schema_list = i.split("|")
        db_id = question_schema_list[1].strip()
        question = question_schema_list[0].strip()
        all_tables_columns = question_schema_list[2:]
        new_tables_columns = []
        query = copy.copy(batch["query"][index])
        query = query.lower()
        query_word = query.split(" ")
        original_word_for_sql_col, synonym_word_for_sql_col = columns_name_dict[db_id]["original"], \
                                                              columns_name_dict[db_id]["synonym"]
        original_table_word, synonym_table_word = tables_name_dict[db_id]["original"], tables_name_dict[db_id][
            "synonym"]
        for query_index, one_query_word in enumerate(query_word):
            one_question_word_split = None
            if "." in one_query_word:
                one_question_word_split = one_query_word.split(".")
            for word_index, pattern in enumerate(original_word_for_sql_col):
                if one_question_word_split == None:
                    match = re.search("^" + pattern, one_query_word.lower())
                    if match:
                        query_word[query_index] = synonym_word_for_sql_col[word_index]
                        # print(query_word[query_index])
                else:
                    split_word = one_question_word_split[1]
                    match = re.search("^" + pattern, split_word.lower())
                    if match:
                        one_question_word_split[1] = synonym_word_for_sql_col[word_index]
                        # print(one_question_word_split[1])
                    one_query_word = ".".join(one_question_word_split)
                    query_word[query_index] = one_query_word

            if one_question_word_split != None:
                for word_index, pattern in enumerate(original_table_word):
                    split_word = one_question_word_split[0]
                    match = re.search("^" + pattern, split_word.lower())
                    if match:
                        one_question_word_split[0] = synonym_table_word[word_index]
                        # print(one_question_word_split[0])
                    one_query_word = ".".join(one_question_word_split)
                    query_word[query_index] = one_query_word
            else:
                for word_index, pattern in enumerate(original_table_word):  # 匹配表名，表名必不包含点(.)
                    match = re.search("^" + pattern, one_query_word.lower())
                    if match:
                        query_word[query_index] = synonym_table_word[word_index]
                        # print(query_word[query_index])

        query = " ".join(query_word)
        query_synonym.append(query)
        flag,instruction_list=False,None
        for tab_index, tab_col_pair in enumerate(all_tables_columns):
            new_column_list = []
            if ":" not in tab_col_pair:
                # print(tab_col_pair)
                # print("*"*20)
                if "columns" in tab_col_pair:
                    instruction_list=tab_col_pair.split(" ")
                    for instruction_index,instruction_seg in enumerate(copy.deepcopy(instruction_list)):
                        if instruction_seg=="contain":
                            break
                        if instruction_seg=="columns":
                            continue
                        col_ins=instruction_seg
                        if col_ins in columns_name_dict[db_id]["original"]:
                            k = columns_name_dict[db_id]["original"].index(col_ins)
                            instruction_list[instruction_index]=columns_name_dict[db_id]["synonym"][k]

                    schema_linking = " ".join(instruction_list)
                    new_tables_columns.append(schema_linking)
                    # print(instruction_list)
                    # print("*" * 20)
                    continue
                else:
                    continue
            tab_col_list=tab_col_pair.split(":")
            if len(tab_col_list)==2:
                table_name, column = tab_col_list
            elif len(tab_col_list)>2:
                table_name = tab_col_pair[0]
                column = ":".join(tab_col_pair[1:])
            else:
                print(tab_col_pair)
                continue
            columns_list = column.split(",")
            table_name = table_name.strip()
            columns_list = [i.strip() for i in columns_list if i != ' ']
            if table_name in tables_name_dict[db_id]["original"]:
                k = tables_name_dict[db_id]["original"].index(table_name)
                new_table_name = tables_name_dict[db_id]["synonym"][k]
            else:
                new_table_name = table_name
            for col_index, col in enumerate(columns_list):
                col_value = None
                # 针对有db_id的情况
                if "(" in col and ")" in col:
                    print(col)
                    match=re.search("(.*?)\((.*)\)", col)
                    if match!=None:
                        col_except_value,value=match.group(1),match.group(2)
                    else:
                        print(col)
                        continue
                    col_except_value, value_bracket = col_except_value.strip(), value.strip()
                    col_value = [col_except_value, value_bracket]
                if col_value == None:
                    if col in columns_name_dict[db_id]["original"]:
                        # print(col)
                        k = columns_name_dict[db_id]["original"].index(col)
                        new_column_list.append(columns_name_dict[db_id]["synonym"][k])
                    else:
                        new_column_list.append(col)
                else:
                    if col_value[0] in columns_name_dict[db_id]["original"]:
                        k = columns_name_dict[db_id]["original"].index(col_value[0])
                        col_value[0] = columns_name_dict[db_id]["synonym"][k]
                        col_value_string = "{column} ( {values} )".format(column=col_value[0], values=col_value[1])
                        new_column_list.append(col_value_string)
                    else:
                        col_value_string = "{column} ( {values} )".format(column=col_value[0], values=col_value[1])
                        new_column_list.append(col_value_string)
            new_column_list=[i for i in new_column_list if i!='']
            # print(new_column_list)
            schema_linking = new_table_name + " : " + " , ".join(new_column_list) + " , "
            # print(schema_linking)
            # schema_linking = new_table_name + " : " + " , ".join(new_column_list)
            new_tables_columns.append(schema_linking)
        question_word = question.split(" ")
        original_word, synonym_word = synonym_dict[db_id]["original"], synonym_dict[db_id]["synonym"]
        for question_index, one_question_word in enumerate(question_word):
            for word_index, pattern in enumerate(original_word):
                match = re.search("^" + pattern, one_question_word.lower())
                if match:
                    if "_" in synonym_word[word_index]:  # 一些近义词word本来就有下划线
                        except_line_word = synonym_word[word_index].split("_")
                        except_line_word = " ".join(except_line_word)
                        question_word[question_index] = except_line_word
                    else:
                        question_word[question_index] = synonym_word[word_index]
        question = " ".join(question_word)
        question_schema_list = [question, db_id]
        question_schema_list.extend(new_tables_columns)
        synonym_inputs.append(" | ".join(question_schema_list))
    return synonym_inputs,query_synonym

def spider_pre_process_function(
    batch: dict,
    max_source_length: Optional[int],
    max_target_length: Optional[int],
    data_training_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizerBase,
    synonym_dict_path:str="./seq2seq/dict.txt",
    use_synonym:bool= False
) -> dict:
    prefix = data_training_args.source_prefix if data_training_args.source_prefix is not None else ""

    inputs = [
        spider_get_input(question=question, serialized_schema=serialized_schema, prefix=prefix,use_instruction=data_training_args.use_instruction)
        for question, serialized_schema in zip(batch["question"], batch["serialized_schema"])
    ]

    if use_synonym:
        synonym_inputs,query_synonym=add_synonym(synonym_dict_path, inputs, batch)
        inputs.extend(synonym_inputs)
        batch["query"].extend(query_synonym)
        batch["db_id"]=batch["db_id"]*2
        #删除多的空格
        inputs_split = [i.split() for i in inputs]
        inputs_split= [[j for j in i if j != ''] for i in inputs_split]
        query_split=[i.split() for i in batch["query"]]
        query_split=[[j for j in i if j != ''] for i in query_split]
        inputs=[" ".join(i) for i in inputs_split]
        batch["query"]=[" ".join(i) for i in query_split]

    model_inputs: dict = tokenizer(
        inputs,
        max_length=max_source_length,
        padding=False,
        truncation=True,
        return_overflowing_tokens=False,
    )

    targets = [
        spider_get_target(
            query=query,
            db_id=db_id,
            normalize_query=data_training_args.normalize_query,
            target_with_db_id=data_training_args.target_with_db_id,
        )
        for db_id, query in zip(batch["db_id"], batch["query"])
    ]

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding=False,
            truncation=True,
            return_overflowing_tokens=False,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


class SpiderTrainer(Seq2SeqTrainer):
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
        if self.target_with_db_id:
            # Remove database id from all predictions
            predictions = [pred.split("|", 1)[-1].strip() for pred in predictions]
        # TODO: using the decoded reference labels causes a crash in the spider evaluator
        # if self.ignore_pad_token_for_loss:
        #     # Replace -100 in the labels as we can't decode them.
        #     label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        # decoded_references = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        # references = [{**{"query": r}, **m} for r, m in zip(decoded_references, metas)]
        references = metas
        return self.metric.compute(predictions=predictions, references=references)
