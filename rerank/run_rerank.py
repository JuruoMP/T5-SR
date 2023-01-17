import sys

sys.path.append("./")
import re
import numpy as np
from rerank.pre_process import tokenizer_process, schema_process
from rerank.sql_ranking_model import RobertaSequenceClassificationForRanking
from datasets import load_dataset
from transformers import RobertaTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader, WeightedRandomSampler
from rerank.model_evaluate import get_best_acc, get_test_sql, get_kfold_acc
import json
import argparse


def eval_kfold(config: dict) -> None:
    tokenizer = RobertaTokenizer.from_pretrained(config["eval_model_path"])
    model = RobertaSequenceClassificationForRanking.from_pretrained(config["eval_model_path"], num_labels=2)

    exact_val_dataset = load_dataset("json", data_files=config["exact_val_path"], field="data")
    exact_val_dataset = exact_val_dataset.map(lambda case: schema_process(case, config["use_foreign_key"]),
                                              batched=False, num_proc=1, load_from_cache_file=False)
    exact_val_dataset = exact_val_dataset["train"].map(
        lambda batch: tokenizer_process(batch, tokenizer, 512),
        batched=True,
        num_proc=1,
        remove_columns=exact_val_dataset["train"].column_names,
        load_from_cache_file=False
    )
    datacollator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        max_length=512,
        pad_to_multiple_of=8
    )

    exact_val_loader = DataLoader(exact_val_dataset, batch_size=config["batch_size"], shuffle=False,
                                  collate_fn=datacollator)

    get_kfold_acc(config, model, exact_val_loader)


def eval(config: dict) -> None:
    tokenizer = RobertaTokenizer.from_pretrained(config["eval_model_path"])
    model = RobertaSequenceClassificationForRanking.from_pretrained(config["eval_model_path"], num_labels=2)

    exact_val_dataset = load_dataset("json", data_files=config["exact_val_path"], field="data")
    exact_val_dataset = exact_val_dataset.map(lambda case: schema_process(case, config["use_foreign_key"]),
                                              batched=False, num_proc=1, load_from_cache_file=False)
    exact_val_dataset = exact_val_dataset["train"].map(
        lambda batch: tokenizer_process(batch, tokenizer, 512),
        batched=True,
        num_proc=1,
        remove_columns=exact_val_dataset["train"].column_names,
        load_from_cache_file=False
    )
    datacollator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        max_length=512,
        pad_to_multiple_of=8
    )

    exact_val_loader = DataLoader(exact_val_dataset, batch_size=config["batch_size"], shuffle=False,
                                  collate_fn=datacollator)

    exact_acc, best_groups, op_tuple = get_best_acc(config, model, exact_val_loader)
    print(exact_acc)
    with open(config["roberta_index_path"], "w") as f:
        json.dump({"data": best_groups}, f, indent=4)
    with open(config["op_path"], "w") as f:
        json.dump({"op": list(op_tuple)}, f, indent=4)
    index = best_groups
    with open(config["exact_val_path"], "r") as f:
        data = json.load(f)["data"]

    sql = []
    for i in index:
        sql.append(data[i]["query"].strip() + "\n")
    with open(config["after_rerank_path"], "w") as f:
        f.writelines(sql)

def concatate_value(pre_list):
    collect_list,pre_list_new = [],[]
    flag, tag = False, False
    for item in pre_list:
        if "'" not in item and '"' not in item and flag == False:
            pre_list_new.append(item)
        elif "'" in item or '"' in item or flag == True:
            denote_len = 0
            if "'" in item:
                denote_len = len([i for i, x in enumerate(list(item)) if x == "'"])
            elif '"' in item:
                denote_len = len([i for i, x in enumerate(list(item)) if x == '"'])
            if denote_len == 2:
                pre_list_new.append(item)
            else:
                if flag == True and denote_len == 1:
                    collect_list.append(item)
                    flag = False
                    tag = True
                else:
                    flag = True
                    collect_list.append(item)
        if flag == False and tag == True:
            pre_list_new.append(" ".join(collect_list))
            tag = False
            collect_list = []
    return pre_list_new

def fix(pre_list,aft_list,pre_denote,aft_denote):

    if len(pre_denote)!=len(aft_denote):
        return pre_list,aft_list
    else:
        for index,item in enumerate(pre_denote):
            aft_list[aft_denote[index]+1]=pre_list[item+1]
        return pre_list,aft_list

def fix_str(pre_list,aft_list,pre_denote,aft_denote):
    if len(pre_denote)==0 or len(pre_denote)!=len(aft_denote) :
        return pre_list,aft_list
    else:
        for index,item in enumerate(pre_denote):
            if "'" in pre_list[item]:
                denote_len=len([i for i, x in enumerate(list(pre_list[item])) if x=="'"])
            elif '"' in pre_list[item]:
                denote_len = len([i for i, x in enumerate(list(pre_list[item])) if x=='"'])
            else:
                print("fix error")
                return pre_list, aft_list
                # raise ValueError("fix error")
            if denote_len==2:
                aft_list[aft_denote[index]] = pre_list[item]
            else:
                print("fix error")
                # raise ValueError("fix error")
                return pre_list, aft_list
        return pre_list,aft_list

def fix_equal(pre_list,aft_list,pre_denote,aft_denote):
    if len(pre_denote)!=len(aft_denote):
        return pre_list,aft_list
    else:
        for index,item in enumerate(pre_denote):
            try:
                pre_num=float(pre_list[item+1])
                aft_num=float(aft_list[aft_denote[index]+1])#验证是否能转换为float
                aft_list[aft_denote[index] + 1] = pre_list[item + 1]

            except:
                if "'" in pre_list[item+1] or '"' in pre_list[item+1]:
                    aft_list[aft_denote[index] + 1] = pre_list[item + 1]
        return pre_list,aft_list

def lev(str_a,str_b):
    str_a=str_a.lower()
    str_b=str_b.lower()
    matrix_ed=np.zeros((len(str_a)+1,len(str_b)+1),dtype=np.int32)
    matrix_ed[0]=np.arange(len(str_b)+1)
    matrix_ed[:,0] = np.arange(len(str_a) + 1)
    for i in range(1,len(str_a)+1):
        for j in range(1,len(str_b)+1):
            dist_1 = matrix_ed[i - 1, j] + 1
            dist_2 = matrix_ed[i, j - 1] + 1
            dist_3 = matrix_ed[i - 1, j - 1] + (1 if str_a[i - 1] != str_b[j - 1] else 0)
            matrix_ed[i,j]=np.min([dist_1, dist_2, dist_3])
    # print(matrix_ed)
    return matrix_ed[-1,-1]

def heuristic_fix(pre_list,aft_list,pre_denote,aft_denote):
    pre_key_list, pre_value_list, aft_key_list, aft_value_list = [], [], [], []
    for index, item in enumerate(pre_denote):
        if item == 0 or item == len(pre_list) - 1:
            return pre_list, aft_list
        pre_key_list.append(pre_list[item - 1])
        pre_value_list.append(pre_list[item + 1])
    for index, item in enumerate(aft_denote):
        if item == 0 or item == len(aft_list) - 1:
            return pre_list, aft_list
        aft_key_list.append(aft_list[item - 1])
        aft_value_list.append(aft_list[item + 1])

    pre_key_norm, aft_key_norm = get_norm_key(pre_key_list), get_norm_key(aft_key_list)
    if pre_key_norm == None or aft_key_norm == None:
        return pre_list, aft_list

    use_tag = [False] * len(pre_list)
    new_aft_value = []
    for aft_index, aft_str in enumerate(aft_key_norm):
        lev_scores = []
        for pre_index, pre_str in enumerate(pre_key_norm):
            score = lev(aft_str, pre_str)
            lev_scores.append(score)
        while (1):
            min_score_index = lev_scores.index(min(lev_scores))
            if use_tag[min_score_index] == False:
                break
            else:
                lev_scores[min_score_index] = 9999
        new_aft_value.append(pre_value_list[min_score_index])
        use_tag[min_score_index] = True
    for index, item in enumerate(new_aft_value):
        aft_list[aft_denote[index] + 1] = item
    return pre_list, aft_list

def new_fix_equal(pre_list, aft_list, pre_denote, aft_denote):
    if len(pre_denote) != len(aft_denote):
        return pre_list, aft_list
    if len(pre_denote)==1:
        pre_item,aft_item=pre_denote[0],aft_denote[0]
        try:
            pre_num = float(pre_list[pre_item + 1])
            aft_num = float(aft_list[aft_item + 1])  # 验证是否能转换为float
            aft_list[aft_item + 1] = pre_list[pre_item + 1]
        except:
            if "'" in pre_list[pre_item + 1] or '"' in pre_list[pre_item + 1]:
                aft_list[aft_item + 1] = pre_list[pre_item + 1]
        return pre_list, aft_list
    else:
        new_pre_denote,new_aft_denote=[],[]
        for index, item in enumerate(pre_denote):
            try:
                pre_num = float(pre_list[item + 1])
                aft_num = float(aft_list[aft_denote[index] + 1])  # convert float?
                new_aft_denote.append(aft_denote[index])
                new_pre_denote.append(item)
                # aft_list[aft_denote[index] + 1] = pre_list[item + 1]
            except:
                if "'" in pre_list[item + 1] or '"' in pre_list[item + 1]:
                    # aft_list[aft_denote[index] + 1] = pre_list[item + 1]
                    new_aft_denote.append(aft_denote[index])
                    new_pre_denote.append(item)
        pre_list, aft_list=heuristic_fix(pre_list,aft_list,new_pre_denote,new_aft_denote)
        return pre_list, aft_list

def new_fix(pre_list,aft_list,pre_denote,aft_denote):
    if len(pre_denote)!=len(aft_denote):
        return pre_list,aft_list
    if len(pre_denote)==1:
        for index, item in enumerate(pre_denote):
            aft_list[aft_denote[index] + 1] = pre_list[item + 1]
        return pre_list, aft_list
    return heuristic_fix(pre_list,aft_list,pre_denote,aft_denote)

def get_norm_key(key_list):
    key_norm=[]
    for index,item in enumerate(key_list):
        if "." in item:
            tab,col=item.split(".")
            key_norm.append(col.strip().lower())
        elif "count(*)" == item.lower():
            key_norm.append(item.strip().lower())
        elif "'" in item or '"' in item:
            key_norm.append(item.strip().lower())
        else:
            return None
    return key_norm

def main(pre_rerank_path,aft_rerank_path,save_path):
    with open(pre_rerank_path, "r") as f:
    # with open("./3B_analyze_c/sql/demo/pre_rerank.sql","r") as f:
        pre_rerank=f.readlines()
    with open(aft_rerank_path,"r") as f:
        aft_rerank=f.readlines()
    pre_rerank=[" ".join(item.split()) for item in pre_rerank]
    aft_rerank=[" ".join(item.split()) for item in aft_rerank]
    pre_rerank=[item.lstrip('"') for item in pre_rerank]
    pre_rerank = [item.lstrip("'") for item in pre_rerank]
    aft_rerank = [item.lstrip('"') for item in aft_rerank]
    aft_rerank = [item.lstrip("'") for item in aft_rerank]
    pre_rerank = [re.sub("count \( \* \)", "count(*)", item) for item in pre_rerank]
    aft_rerank = [re.sub("count \( \* \)", "count(*)", item) for item in aft_rerank]
    new_aft_rerank=[]
    for index,pre_item in enumerate(pre_rerank):
        pre_list=pre_item.strip().split(" ")
        aft_list=aft_rerank[index].strip().split(" ")
        pre_list = concatate_value(pre_list)
        aft_list = concatate_value(aft_list)
        pre_denote=[i for i ,x in enumerate(pre_list) if x==">="]
        aft_denote=[i for i ,x in enumerate(aft_list) if x==">="]
        pre_list,aft_list=new_fix(pre_list,aft_list,pre_denote,aft_denote)
        pre_denote = [i for i, x in enumerate(pre_list) if x == "<="]
        aft_denote = [i for i, x in enumerate(aft_list) if x == "<="]
        pre_list, aft_list = new_fix(pre_list, aft_list, pre_denote, aft_denote)
        pre_denote = [i for i, x in enumerate(pre_list) if x == ">"]
        aft_denote = [i for i, x in enumerate(aft_list) if x == ">"]
        pre_list, aft_list = new_fix(pre_list, aft_list, pre_denote, aft_denote)
        pre_denote = [i for i, x in enumerate(pre_list) if x == "<"]
        aft_denote = [i for i, x in enumerate(aft_list) if x == "<"]
        pre_list, aft_list = new_fix(pre_list, aft_list, pre_denote, aft_denote)
        pre_denote = [i for i, x in enumerate(pre_list) if x == "="]
        aft_denote = [i for i, x in enumerate(aft_list) if x == "="]
        pre_list, aft_list = new_fix_equal(pre_list, aft_list, pre_denote, aft_denote)
        pre_denote = [i for i, x in enumerate(pre_list) if '"' in x and pre_list[i-1] not in ['>=','<=','>',"<","="]]
        aft_denote = [i for i, x in enumerate(aft_list) if '"' in x and aft_list[i-1] not in ['>=','<=','>',"<","="]]
        pre_list, aft_list = fix_str(pre_list, aft_list, pre_denote, aft_denote)
        pre_denote = [i for i, x in enumerate(pre_list) if "'" in x and pre_list[i-1] not in ['>=','<=','>',"<","="]]
        aft_denote = [i for i, x in enumerate(aft_list) if "'" in x and aft_list[i-1] not in ['>=','<=','>',"<","="]]
        pre_list, aft_list = fix_str(pre_list, aft_list, pre_denote, aft_denote)
        new_aft_rerank.append(" ".join(aft_list) + "\n")

    with open(save_path, "w") as f:
    # with open("./3B_analyze_c/sql/demo/new_rerank.sql","w") as f:
        f.writelines(new_aft_rerank)

def get_pre_rerank(prediction_path,pre_rerank_path):
    with open(prediction_path,"r") as f:
        prediction_data=f.readlines()
    best_t5_sql=[]
    for index,item in enumerate(prediction_data):
        prediction_sql=item
        if index%10==0:
            best_t5_sql.append(prediction_sql)
    with open(pre_rerank_path,"w") as f:
        f.writelines(best_t5_sql)

def test(config: dict) -> None:
    tokenizer = RobertaTokenizer.from_pretrained(config["test_model_path"])
    model = RobertaSequenceClassificationForRanking.from_pretrained(config["test_model_path"], num_labels=2)

    exact_test_dataset = load_dataset("json", data_files=config["exact_test_path"], field="data")
    exact_test_dataset = exact_test_dataset.map(lambda case: schema_process(case, config["use_foreign_key"]),
                                                batched=False, num_proc=1, load_from_cache_file=False)
    exact_test_dataset = exact_test_dataset["train"].map(
        lambda batch: tokenizer_process(batch, tokenizer, 512),
        batched=True,
        num_proc=1,
        remove_columns=exact_test_dataset["train"].column_names,
        load_from_cache_file=False
    )
    datacollator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        max_length=512,
        pad_to_multiple_of=8,
    )

    exact_test_loader = DataLoader(exact_test_dataset, batch_size=config["batch_size"], shuffle=False,
                                   collate_fn=datacollator)
    get_test_sql(config, model, test_loader=exact_test_loader)

    get_pre_rerank(config["prediction_path"],config["pre_rerank_path"])
    main(config["pre_rerank_path"],config["after_rerank_path"],config["save_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval or test')
    parser.add_argument('-t', '--type', default='eval', type=str, help='eval/test')
    args = parser.parse_args()

    if args.type == "eval":
        config = {
            "exact_val_path": "logdir/data/exact_eval_data.json",
            "accumulation_steps": 1,
            "batch_size": 32,
            "num_labels": 2,
            "is_save": True,
            "use_foreign_key": True,
            "use_soft": True,
            "beam_num": 10,
            "use_balance": False,
            "t5_score_path": "logdir/data/t5_scores.txt",
            "roberta_index_path": "logdir/data/roberta_index.txt",
            "eval_model_path": "checkpoint_rerank",
            "after_rerank_path": "logdir/data/after_rerank.sql",
            "op_path": "static/op.json",
            "use_record": False
        }
        eval(config)
        # eval_kfold(config)


    elif args.type == "test":
        config = {
            "exact_test_path": "logdir/data/exact_dev_data.json",
            "batch_size": 32,
            "num_labels": 2,
            "is_save": True,
            "use_foreign_key": True,
            "use_soft": True,
            "beam_num": 10,
            "use_balance": False,
            "t5_score_path": "logdir/data/t5_scores.txt",
            "test_model_path": "checkpoint_rerank",
            "after_rerank_path": "logdir/data/after_rerank.sql",
            "op_path": "static/op.json",
            "prediction_path":"logdir/data/dev_beam_result.sql",
            "pre_rerank_path":"logdir/data/pre_rerank.sql",
            "save_path":"logdir/data/after_rerank.sql"
        }
        test(config)