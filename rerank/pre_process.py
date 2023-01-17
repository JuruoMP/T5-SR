# import sys
# import os
# os.chdir("../")
# sys.path.insert(0, './')
import numpy
import re
import torch
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel, AdamW
from rerank.sql_ranking_model import RobertaSequenceClassificationForRanking
from datasets import load_dataset
from rerank.collator import DataCollatorWithPadding
from torch.utils.data import DataLoader
import copy


# import torch
def normalize(query: str) -> str:  # sql处理
    def comma_fix(s):
        # Remove spaces in front of commas
        return s.replace(" , ", ", ")

    def white_space_fix(s):
        # Remove double and triple spaces
        return " ".join(s.split())

    def lower(s):
        # Convert everything except text between (single or double) quotation marks to lower case
        return re.sub(r"\b(?<!['\"])(\w+)(?!['\"])\b", lambda match: match.group(1).lower(), s)

    return comma_fix(white_space_fix(lower(query)))


def schema_process(example, use_foregin_key=True, tmp=0.1):
    label = example["label"]
    # 调整软标签
    for index, i in enumerate(copy.copy(label)):
        if i == 0.3:
            label[index] = label[index] + tmp
        elif i == 0.7:
            label[index] = label[index] - tmp
        else:
            break

    query = example["query"].lower()
    query = normalize(query)
    question = example["question"].lower()
    if use_foregin_key:
        tables = example["table_with_foreign_keys"].lower().split("/")
        # print(tables)
    else:
        tables = example["table"].lower().split("/")
    tables = [i.strip() for i in tables if i != '']
    schema_linking = ""
    # print(tables)
    query_list = query.split(" ")
    query_list = [re.sub("\(", "", i) for i in query_list]
    query_list = [re.sub("\)", "", i) for i in query_list]
    query_list = [re.sub("\*", "", i) for i in query_list]
    query_list = [i for i in query_list if i != '']
    # print(query_list)
    for table in tables:
        if table == '':
            continue
        # print(table)
        table_split = table.split(":")
        if len(table_split) == 2:
            table_name, col_str = table_split
        elif len(table_split) > 2:
            table_name, col_str = table_split[0], ":".join(table_split[1:])
        else:
            raise ValueError("table split error")
        if table_name.strip() in query_list:
            name_index = query_list.index(table_name.strip())
            if query_list[name_index - 1].strip() not in ["from", "join"]:
                continue
            schema_linking += " | "
            schema_linking += table

    input_str = question.strip() + " " + schema_linking.strip() + " " + query
    # print( schema_linking.strip() )
    input_str = " ".join(input_str.split())
    # print(input_str)
    # print(label)
    return {"input_str": input_str, "labels": label}


def tokenizer_process(batch, tokenizer, max_source_length):  # 调用时要使用匿名函数
    inputs = batch["input_str"]
    model_inputs: dict = tokenizer(
        inputs,
        max_length=max_source_length,
        padding=False,
        truncation=True,
        return_overflowing_tokens=False,
    )
    model_inputs["labels"] = numpy.array((batch["labels"]))
    # model_inputs["labels"] = 0
    return model_inputs


if __name__ == "__main__":
    # path= r"./data/demo.json"
    path = r"../3B_data/db_content/dev_data_db_content.json"
    dataset = load_dataset('json', data_files=path, field='data')
    dataset = dataset.map(schema_process,
                          batched=False, num_proc=1,
                          load_from_cache_file=False)
    tokenizer = RobertaTokenizer.from_pretrained("../roberta-large")

    model = RobertaSequenceClassificationForRanking.from_pretrained("../roberta-large", num_labels=2)

    # # print(dataset.column_names)
    dataset = dataset["train"].map(
        lambda batch: tokenizer_process(batch, tokenizer, 512),
        batched=True,
        num_proc=8,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False  # not data_training_args.overwrite_cache,
    )
    # labels=[]
    # for i in dataset:
    #     labels.append(i["labels"])
    #     print(i["labels"])
    # labels=torch.tensor(labels)
    # class_sample_count = torch.tensor(
    #     [(labels== t).sum() for t in torch.unique(labels, sorted=True)])
    # weight = 1. / class_sample_count.float()
    # samples_weight = torch.tensor([weight[t["labels"]] for t in dataset])
    # print(samples_weight)
    # ignore_pad_token_for_loss = True
    datacollator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        max_length=512,
        pad_to_multiple_of=8
    )
    # #
    # # # print(dataset)
    trainLoader1 = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=datacollator)
    for i in trainLoader1:
        print(i)
    # trainLoader2 = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=datacollator)
    # sum_out = 0
    # time = 0
    # print(dataset[0])

    # print(list(model.named_parameters()))

    # pa1,pa2=[],[]
    # for i in model.named_parameters():
    #     # print(i[0])
    #     if "classifier" in i[0]:
    #         pa1.append(i[1])
    #     else:
    #         pa2.append(i[1])
    # opt=AdamW([{'params': pa1, 'lr': 1e-5},{'params':pa2}], lr=1e-6, weight_decay=1e-3)
    # print(opt.param_groups)
