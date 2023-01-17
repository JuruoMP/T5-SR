import json
import copy
import argparse


def for_exact_eval(schema_data_path: str, labels_path: str, save_path: str, beam_path: str, beam_num: int = 10):
    pos_hard, pos_soft, neg_num = 0, 0, 0
    with open(schema_data_path, "r") as f:
        schema_data = json.load(f)["data"]
    with open(labels_path, "r") as f:
        labels = f.readlines()
    with open(beam_path, "r") as f:
        beam_data = f.readlines()

    assert len(beam_data) == len(labels), "length error"
    labels = [i.strip() for i in labels]

    data, one_group = [], []
    summ, num_error = 0, 0
    for index, row_beam in enumerate(beam_data):
        row_beam = " ".join(row_beam.strip().split())
        summ += int(labels[index] == "True")

        schema_row = copy.deepcopy(schema_data[int(index / beam_num)])
        one_group.append(row_beam)
        row_label = int(labels[index] == "True")
        if row_label == 1:
            schema_row["query"] = row_beam
            if index % beam_num == 0:
                schema_row["label"] = [0.0, 1.0]
                pos_soft += 1
            else:
                schema_row["label"] = [0.0, 1.0]
                pos_hard += 1
        else:
            schema_row["query"] = row_beam
            schema_row["label"] = [1.0, 0.0]
            neg_num += 1
        data.append(schema_row)

    with open(save_path, "w") as f:
        json.dump({"data": data}, fp=f, indent=4)


def for_exact_test(schema_data_path: str, save_path: str, beam_path: str, beam_num: int = 10):
    with open(schema_data_path, "r") as f:
        schema_data = json.load(f)["data"]
    with open(beam_path, "r") as f:
        beam_data = f.readlines()

    data, one_group = [], []
    for index, row_beam in enumerate(beam_data):
        row_beam = " ".join(row_beam.strip().split())

        schema_row = copy.deepcopy(schema_data[int(index / beam_num)])
        one_group.append(row_beam)
        schema_row["query"] = row_beam
        schema_row["label"] = [-1, -1]
        data.append(schema_row)

    with open(save_path, "w") as f:
        json.dump({"data": data}, fp=f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval or test')
    parser.add_argument('-t', '--type', default='eval', type=str, help='eval/test')
    args = parser.parse_args()

    if args.type == "eval":
        schema_data_path = r"logdir/data/schema_dev.json"
        labels_path = r"logdir/data/dev_beam_label.txt"
        save_path = r"logdir/data/exact_eval_data.json"
        beam_path = r"logdir/data/dev_beam_result.sql"
        for_exact_eval(schema_data_path, labels_path, save_path, beam_path, beam_num=10)
    elif args.type == "test":
        schema_data_path = r"logdir/data/schema_dev.json"
        save_path = r"logdir/data/exact_dev_data.json"
        beam_path = r"logdir/data/dev_beam_result.sql"
        for_exact_test(schema_data_path, save_path, beam_path, beam_num=10)
