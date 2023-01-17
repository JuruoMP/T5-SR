import sys

sys.path.append("./")
import json
from seq2seq.lf_util.sql_dict_parser import SqlParser
from seq2seq.lf_util.sql_fix_parser import SqlFixParser
from rerank.model_evaluate import evaluate_pairs
from nltk import data
import argparse
import os
import traceback


def ssql2sql(config):
    print("run ssql2sql")
    # (sql_parser.table_path, sql_parser.db_dir)
    sql_parser = SqlParser('data/spider/tables.json', db_dir='data/database')
    # sql_parser = SqlParser('', db_dir='')
    parser = SqlFixParser(sql_parser)
    with open(config["prediction_path"]) as f:
        data = json.load(f)
    ssql, sql = [], []
    for index, i in enumerate(data):
        ssql.append(i["prediction"])
    for index, i in enumerate(ssql):
        try:
            db_id, one_ssql = i.split("|")
            ssql_dict = parser.sql_to_dict(db_id.strip(), one_ssql.strip())
            one_sql = parser.dict_to_raw_sql(db_id.strip(), ssql_dict)
            if one_sql.strip() == "":
                one_sql = "error"
            sql.append(one_sql + "\n")
        except:
            # print(traceback.format_exc())
            sql.append("error\n")
    # print(sql)
    with open(config["save_path"], "w") as f:
        f.writelines(sql)

    with open(config["prediction_path"], "r") as f:
        data = json.load(f)
    t5_socres = []
    for index, i in enumerate(data):
        score = float(i["score"])
        t5_socres.append(score)
    with open(config["t5_scores_path"], "w") as f:
        json.dump(t5_socres, f, indent=4)


def get_beam_label(config):
    print("get label")
    with open(config["gold_file_path"], "r") as f:
        a = f.readlines()
    with open(config["predict_file_path"], "r") as f:
        b = f.readlines()
    assert len(a) == len(b)

    labels = evaluate_pairs(config["gold_file_path"], config["predict_file_path"], config["db_dir"], config["table"])
    with open(config["predict_file_path"], "r") as f:
        predict_data = f.readlines()
    _labels = []
    for index, item in enumerate(predict_data):
        _labels.append(str(labels[index]) + "\n")
    with open(config["label_save_path"], "w") as f:
        f.writelines(_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval or test')
    parser.add_argument('-t', '--type', default='eval', type=str, help='eval/test')
    args = parser.parse_args()
    if os.path.exists("logdir/files") == False:
        os.makedirs("logdir/files")
    if args.type == "eval":
        config = {
            "prediction_path": "logdir/spider_rerank_eval/predictions_eval_None.json",
            "save_path": "logdir/data/dev_beam_result.sql",
            "t5_scores_path": "logdir/data/t5_scores.txt",
            "gold_file_path": r"logdir/data/repeat_dev_gold.sql",
            "db_dir": "data/database",
            "table": "data/spider/tables.json",
            "predict_file_path": r"logdir/data/dev_beam_result.sql",
            "label_save_path": "logdir/data/dev_beam_label.txt"
        }
        ssql2sql(config)
        get_beam_label(config)
    elif args.type == "test":
        config = {
            "prediction_path": "logdir/spider_rerank_test/predictions_eval_None.json",
            "save_path": "logdir/data/dev_beam_result.sql",
            "t5_scores_path": "logdir/data/t5_scores.txt",
        }
        ssql2sql(config)
