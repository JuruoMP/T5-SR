import os
import json
import copy
import re
import sys
import copy


def get_schema_data(train_spider_path, table_path, save_path):
    with open(train_spider_path, "r") as f:
        spider_data = json.load(f)
    with open(table_path, "r") as f:
        table_data = json.load(f)

    all_query = []
    for index, item in enumerate(spider_data):
        one_query = {}
        one_query["db_id"] = item["db_id"]
        one_query["query"] = str(item["query"])
        one_query["question"] = item["question"]

        for i, db in enumerate(table_data):
            if one_query["db_id"] == db["db_id"]:
                table_names = db["table_names_original"]
                all_tables = ""
                for j, table in enumerate(table_names):
                    all_tables += table + " : "
                    for k, column in enumerate(db["column_names_original"]):
                        if (column[0] == j):
                            all_tables += column[1].lower() + " & " + db["column_types"][k] + " , "
                    all_tables += "/"
                one_query["table"] = all_tables
                # one_query["foreign_keys"] = db["foreign_keys"]

                column_name_original_copy = copy.deepcopy(db["column_names_original"])
                for i in db["foreign_keys"]:
                    foregin_table_0 = db["table_names_original"][db["column_names_original"][i[1]][0]]
                    foregin_table_1 = db["table_names_original"][db["column_names_original"][i[0]][0]]
                    column_name_original_copy[i[0]][1] = db["column_names_original"][i[0]][
                                                             1] + " ( " + foregin_table_0 + " . " + \
                                                         db["column_names_original"][i[1]][1] + " ) "
                    column_name_original_copy[i[1]][1] = db["column_names_original"][i[1]][
                                                             1] + " ( " + foregin_table_1 + " . " + \
                                                         db["column_names_original"][i[0]][1] + " ) "

                all_tables = ""
                for j, table in enumerate(table_names):
                    all_tables += table + " : "
                    one_table = []  # one_table存储该table所有的column,column_type
                    for k, column in enumerate(column_name_original_copy):
                        if (column[0] == j):
                            all_tables += column[1].lower() + " & " + db["column_types"][k] + " , "
                    all_tables += "/"
                one_query["table_with_foreign_keys"] = all_tables
                break

        all_query.append(one_query)

    # print(all_query[-1])
    # print(len(all_query))
    with open(save_path, "w") as f:
        # json.dump({"data":all_query},f,indent=4)
        json.dump({"data": all_query}, f, indent=4)


if __name__ == "__main__":
    train_spider_path = r"data/spider/dev.json"
    table_path = r"data/spider/tables.json"
    os.makedirs('logdir/data', exist_ok=True)
    save_path = r"logdir/data/schema_dev.json"
    get_schema_data(train_spider_path, table_path, save_path)
