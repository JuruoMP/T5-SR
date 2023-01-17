import logging
import os
import spacy
from nltk.stem import WordNetLemmatizer
import sqlite3
import editdistance

from seq2seq.lf_util.ssql.ssql_parser import SqlStrParser, SqlParser, WHERE_OPS, COND_OPS, AGG_OPS, UNIT_OPS


class SqlStrFixParser(SqlStrParser):
    def __init__(self, table_path, db_dir):
        super().__init__(table_path)
        self.db_dir = db_dir
        self.nlp_model = spacy.load("en_core_web_lg")
        self.lemmatizer = WordNetLemmatizer()
        self.db_cache = {}

    def parse(self, db_id, sql):
        self.current_table = self.tables[db_id]
        try:
            self.db_content, self.inversed_db_content, self.col_types = \
                self.load_db_content(os.path.join(self.db_dir, db_id, f'{db_id}.sqlite'))
        except:
            self.db_content = self.inversed_db_content = self.col_types = None
        tokens = self._tokenize_sql(sql)
        sql_dict, _ = self._parse_sql(tokens, 0)
        sql_dict = self._heuristic_from(sql_dict)
        return sql_dict

    def _parse_col(self, tokens, start_idx):
        idx = start_idx
        is_block = False

        if tokens[idx] == '(':
            is_block = True
            idx += 1
        if tokens[idx] == '*':
            col_column_id = 0
        else:
            col_table_name, col_column_name = tokens[idx].split('.')
            all_table_names = self.current_table['table_names_original']
            all_column_names = self.current_table['column_names_original']
            col_column_id = -100
            for column_id in range(len(all_column_names)):
                table_id, column_name = all_column_names[column_id]
                table_name = all_table_names[table_id]
                if table_name.lower() == col_table_name and column_name.lower() == col_column_name:
                    col_column_id = column_id
                    break
            if col_column_id == -100:
                if col_table_name.lower() in [x.lower() for x in all_table_names]:  # same table name with different column name
                    col_table_id = [x.lower() for x in all_table_names].index(col_table_name.lower())
                    all_column_similarities = []
                    for column_id in range(1, len(all_column_names)):
                        table_id, column_name = all_column_names[column_id]
                        if table_id == col_table_id:
                            all_column_similarities.append((column_id, self.calc_similarity(column_name, col_column_name)))
                    best_similarity = sorted(all_column_similarities, key=lambda x: x[1], reverse=True)[0]
                    col_column_id = best_similarity[0]
                elif col_column_name.lower() in [x[1].lower() for x in all_column_names]:  # same column name with different table name
                    col_candidates = [(i, x) for i, x in enumerate(all_column_names) if x[1].lower() == col_column_name.lower()]
                    all_table_similarities = []
                    for column_id, (table_id, col_name) in col_candidates:
                        table_name = all_table_names[table_id]
                        all_table_similarities.append((column_id, self.calc_similarity(table_name, col_table_name)))
                    best_similarity = sorted(all_table_similarities, key=lambda x: x[1], reverse=True)[0]
                    col_column_id = best_similarity[0]
                else:  # match for neither table nor column
                    all_column_similarities = []
                    for column_id, (table_id, column_name) in enumerate(all_column_names):
                        table_name = all_table_names[table_id]
                        col_name, name = col_table_name + '_' + col_column_name, table_name + '_' + column_name
                        all_column_similarities.append((column_id, self.calc_similarity(col_name, name)))
                    best_similarities = sorted(all_column_similarities, key=lambda x: x[1], reverse=True)[0]
                    col_column_id = best_similarities[0]
        idx += 1
        if is_block:
            idx += 1
        return col_column_id, idx

    def _parse_table(self, table_name):
        table_unit = -100
        all_table_names = self.current_table['table_names_original']
        all_table_similarities = []
        for table_id in range(len(all_table_names)):
            schema_table_name = all_table_names[table_id]
            if schema_table_name.lower() == table_name:
                table_unit = table_id
                break
        if table_unit == -100:  # not found
            for table_id in range(len(all_table_names)):
                schema_table_name = all_table_names[table_id]
                all_table_similarities.append((table_id, self.calc_similarity(schema_table_name.lower(), table_name)))
            best_similarity = sorted(all_table_similarities, key=lambda x: [1], reverse=True)[0]
            table_unit = best_similarity[0]
        return table_unit

    def calc_similarity(self, name1, name2):
        word_map = [('avg', 'average')]
        name1, name2 = name1.lower(), name2.lower()
        vec_sim = self.nlp_model(name1).similarity(self.nlp_model(name2))
        tok1, tok2 = name1.strip().split(), name2.strip().split()
        tok_match = len(set(tok1) & set(tok2))
        tok1 = [self.lemmatizer.lemmatize(x) for x in name1.strip().split('_')]
        tok2 = [self.lemmatizer.lemmatize(x) for x in name2.strip().split('_')]
        tok_lemma_match = len(set(tok1) & set(tok2))
        return tok_match + tok_lemma_match + vec_sim

    def _parse_condition(self, tokens, start_idx):
        idx = start_idx
        conds = []
        if tokens[idx] in COND_OPS:
            conds.append(tokens[idx])
        while idx < len(tokens):
            val_unit_column_name = tokens[idx]
            val_unit, idx = self._parse_val_unit(tokens, idx)
            not_op = False
            if tokens[idx] == 'not':
                not_op = True
                idx += 1
            assert idx < len(tokens) and tokens[idx] in WHERE_OPS, \
                "Error condition: idx: {}, tok: {}".format(idx, tokens[idx])
            op_id = WHERE_OPS.index(tokens[idx])
            idx += 1
            if op_id == WHERE_OPS.index('between'):  # between..and... special case: dual values
                val1, idx = self._parse_value(tokens, idx)
                assert tokens[idx] == 'and'
                idx += 1
                val2, idx = self._parse_value(tokens, idx)
            else:  # normal case: single value
                val1, idx = self._parse_value(tokens, idx)
                val2 = None
            val_unit, val1, val2 = self._val_match_fix(val_unit, val1, val2, pred_col_name=val_unit_column_name)
            conds.append([not_op, op_id, val_unit, val1, val2])
            if idx < len(tokens) and (tokens[idx] in self.CLAUSE_KEYWORDS or tokens[idx] in (")", ";")):
                break
            if idx < len(tokens) and tokens[idx] in COND_OPS:
                conds.append(tokens[idx])
                idx += 1  # skip and/or
        return conds, idx

    def _parse_value(self, tokens, start_idx):
        idx = start_idx
        is_block = False
        if tokens[idx] == '(':
            is_block = True
            idx += 1

        if tokens[idx] == 'select':
            val, idx = self._parse_sql(tokens, idx)
        elif '"' in tokens[idx]:
            val = tokens[idx]
            idx += 1
        else:
            try:
                val = float(tokens[idx])
                if str(val) != tokens[idx]:
                    val = int(tokens[idx])
                idx += 1
            except:
                end_idx = idx
                while end_idx < len(tokens) and \
                        tokens[end_idx] not in (',', ')', 'and') + self.CLAUSE_KEYWORDS + tuple(COND_OPS):
                    end_idx += 1
                if tokens[start_idx] in AGG_OPS and tokens[end_idx] == ')':
                    end_idx += 1
                val, idx = self._parse_col_unit(tokens[start_idx: end_idx], 0)
                idx = end_idx

        if is_block:
            idx += 1
        return val, idx

    def _val_match_fix(self, val_unit, val1, val2, pred_col_name):
        return val_unit, val1, val2
        if isinstance(val1, dict):
            return val_unit, val1, val2
        if self.db_content is None:
            return val_unit, val1, val2

        def col_id_to_name(col_id):
            target_table_id, target_col_name = self.current_table['column_names_original'][col_id]
            target_table_name = self.current_table['table_names_original'][target_table_id]
            return f'{target_table_name}.{target_col_name}'

        def name_to_col_id(name):
            target_table_name, target_col_name = name.split('.')
            target_table_id = self.current_table['table_names_original'].index(target_table_name)
            for col_id, (table_id, col_name) in enumerate(self.current_table['column_names_original']):
                if table_id == target_table_id and col_name == target_col_name:
                    return col_id
            assert False

        unit_op, col_unit1, col_unit2 = val_unit
        assert col_unit2 is None
        agg_id, col_id, is_distinct = col_unit1

        if isinstance(val1, int) or isinstance(val1, float):  # do not process values
            col_name = col_id_to_name(col_id)
            if '*' in col_name:
                return val_unit, val1, val2
            col_type = self.col_types[col_name]
            if col_type != 'number':
                logging.warning('Warning: Text column got number value')
            return val_unit, val1, val2

        if not(isinstance(val1, str) and val1[0] in ('"', '\'')):
            assert isinstance(val1, str) and val1[0] in ('"', '\'')
        val = val1[1:-1]
        if val in self.db_content[col_id_to_name(col_id)]:  # column-value match
            return val_unit, val1, val2
        elif val in self.inversed_db_content:  # value found, match to column
            if len(self.inversed_db_content[val]) > 1:
                logging.warning('Warning: value matched with several columns')
            target_name_candidates = self.inversed_db_content[val]
            # option1: select best matched column
            target_similarities = [(name, self.calc_similarity(name, pred_col_name)) for name in target_name_candidates]
            target_name = sorted(target_similarities, key=lambda x: x[1], reverse=True)[0][0]
            # option2: select the first appeared column
            # target_name = self.inversed_db_content[val][0]  # todo: value maps to several columns?
            new_col_id = name_to_col_id(target_name)
            new_val_unit = (unit_op, (agg_id, new_col_id, is_distinct), col_unit2)
            return new_val_unit, val1, val2
        else:
            for key in self.inversed_db_content:
                if isinstance(key, str) and key.lower() == val.lower():
                    val1 = '"' + key + '"'
                    return val_unit, val1, val2
            edit_distance_list = []
            for key in self.inversed_db_content:
                if isinstance(key, str):
                    edit_distance_list.append((key, editdistance.eval(key.lower(), val.lower())))
            modified_val, best_edit_distance = sorted(edit_distance_list, key=lambda x: x[1])[0]
            if best_edit_distance <= 2:
                val1 = '"' + modified_val + '"'
            return val_unit, val1, val2

    def load_db_content(self, db_file):
        if db_file in self.db_cache:
            return self.db_cache[db_file]
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("select name from sqlite_master where type='table'")
        tables = cursor.fetchall()

        db_dict, col_types = {}, {}
        for (table,) in tables:
            table_dict = {}
            col_index_dict = {}
            cursor.execute(f"PRAGMA table_info({table})")
            col_names = []
            for col_info in cursor.fetchall():
                col_name = col_info[1]
                col_names.append(col_name)
                if col_info[2].startswith('varchar'):
                    col_types[f'{table}.{col_name}'] = 'text'
                else:
                    col_types[f'{table}.{col_name}'] = 'number'
            for col_idx, col_name in enumerate(col_names):
                col_index_dict[col_idx] = col_name
                table_dict[col_name] = []
            cursor.execute(f"select * from {table}")
            for x in cursor.fetchall():
                for col_index, d in enumerate(x):
                    col_name = col_index_dict[col_index]
                    if d not in table_dict[col_name]:
                        table_dict[col_name].append(d)
            db_dict[table] = table_dict

        data_list = {}
        for table_name, table_dict in db_dict.items():
            for col_name, col_data in table_dict.items():
                data_list[f'{table_name}.{col_name}'] = col_data

        inversed_data_dict = {}
        for key, val_list in data_list.items():
            for val in val_list:
                l = inversed_data_dict.get(val, [])
                l.append(key)
                inversed_data_dict[val] = l

        self.db_cache[db_file] = (data_list, inversed_data_dict, col_types)
        return data_list, inversed_data_dict, col_types


class SqlFixParser(SqlParser):
    def __init__(self, sql_parser: SqlParser):
        super().__init__(sql_parser.table_path, sql_parser.db_dir)
        self.sql_str_parser = SqlStrFixParser(self.table_path, self.db_dir)


if __name__ == '__main__':
    from tqdm import tqdm
    pred_log = [x.strip().split('\t') for x in open('logdir/spider_log/pred.txt').readlines()]
    sql_parser = SqlParser('data/spider/tables.json', 'data/database')
    sql_fix_parser = SqlFixParser(sql_parser)
    all_ori_status_list = []
    all_status_list = []

    with open('fix_log.txt', 'w') as fw:
        correct, wrong, total, illegal = 0, 0, 0, 0
        exec_correct, exec_wrong = 0, 0
        for i, (db_name, qm_status, em_status, pred_sql, gold) in tqdm(enumerate(pred_log[:])):
            # if i in (159,507,698,940):
            #     continue
            all_ori_status_list.append((qm_status, em_status))
            try:
                pred_sql_dict = sql_fix_parser.sql_to_dict(db_name, pred_sql)
                pred_fix_sql = sql_fix_parser.dict_to_sql(db_name, pred_sql_dict)
            except:
                pred_sql_dict = {}
                pred_fix_sql = ''
                illegal += 1
            try:
                # gold_sql_dict = sql_parser.sql_to_dict(db_name, gold)
                # gold_raw_sql = sql_parser.dict_to_raw_sql(db_name, gold_sql_dict)
                gold_raw_sql = gold
            except:
                gold_raw_sql = ''
            if True:
                pred_raw_sql = sql_parser.dict_to_raw_sql(db_name, pred_sql_dict)
                if sql_parser.check_equal_script(db_name, pred_raw_sql, gold_raw_sql):
                    fw.write(f'Correct\t{gold}\t{pred_sql}\t{pred_fix_sql}\n')
                    all_status_list.append('correct')
                    correct += 1
                else:
                    fw.write(f'Wrong\t{gold}\t{pred_sql}\t{pred_fix_sql}\n')
                    all_status_list.append('wrong')
                    wrong += 1
            else:
                fw.write(f'Wrong\t{gold}\t{pred_sql}\t{pred_fix_sql}\n')
                wrong += 1

            try:
                pred_sql_dict = sql_parser.raw_sql_to_dict(db_name, pred_raw_sql)
                exec_match = sql_parser.check_exec_match_script(
                    db_name, pred_raw_sql, gold_raw_sql, pred_sql_dict,
                    sql_parser.raw_sql_to_dict(db_name, gold_raw_sql))
                if exec_match is True:
                    exec_correct += 1
                else:
                    exec_wrong += 1
            except:
                exec_wrong += 1
            total += 1

    print(f'{correct} / {total} = {correct / total}')
    print(f'{exec_correct} / {total} = {exec_correct / total}')
    print(illegal)

    with open('predict.txt', 'w') as fw:
        for db_name, qm_status, em_status, pred_sql, gold in tqdm(pred_log):
            try:
                pred_sql_dict = sql_fix_parser.sql_to_dict(db_name, pred_sql)
                pred_fix_sql = sql_fix_parser.dict_to_sql(db_name, pred_sql_dict)
            except:
                pred_sql_dict = {}
                pred_fix_sql = 'select count(*) from singer'
                illegal += 1
            pred_raw_sql = sql_parser.dict_to_raw_sql(db_name, pred_sql_dict)
            if not pred_raw_sql:
                pred_raw_sql = 'select count(*) from singer'
            fw.write(f'{pred_raw_sql}\t{db_name}\n')

