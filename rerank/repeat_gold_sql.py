import os
import copy

if __name__ == "__main__":
    if os.path.exists("logdir/data") == False:
        os.makedirs("logdir/data")
    dev_gold_path = "spider/dev.json"
    save_path = "logdir/data/repeat_dev_gold.txt"
    with open(dev_gold_path, "r") as f:
        sql_data = f.readlines()
    repeat_data = []
    for i in sql_data:
        for j in range(10):  # beam num
            repeat_data.append(copy.copy(i))
    with open(save_path, "w") as f:
        f.writelines(repeat_data)
