import spacy
from nltk.stem import WordNetLemmatizer
import editdistance

from seq2seq.lf_util.sql_dict_parser import SqlStrParser, SqlParser


class SqlStrFixParser(SqlStrParser):
    def __init__(self, table_path):
        super().__init__(table_path)
        self.nlp_model = spacy.load("en_core_web_lg")
        self.lemmatizer = WordNetLemmatizer()

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
                        if table_id != col_table_id:
                            continue
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


class SqlFixParser(SqlParser):
    def __init__(self, sql_parser: SqlParser):
        super().__init__(sql_parser.table_path, sql_parser.db_dir)
        self.sql_str_parser = SqlStrFixParser(self.table_path)


if __name__ == '__main__':
    from tqdm import tqdm
    pred_log = [x.strip().split('\t') for x in open('logdir/spider_log/pred.txt').readlines()]
    sql_parser = SqlParser('data/spider/tables.json', 'data/database')
    sql_fix_parser = SqlFixParser(sql_parser)

    correct, wrong, total = 0, 0, 0
    for db_name, status, pred_sql, gold in tqdm(pred_log[:]):
        try:
            pred_sql_dict = sql_fix_parser.sql_to_dict(db_name, pred_sql)
            pred_raw_sql = sql_parser.dict_to_raw_sql(db_name, pred_sql_dict)
            pred_fix_sql = sql_parser.dict_to_sql(db_name, pred_sql_dict)
        except:
            pred_raw_sql = ''
        if sql_parser.check_equal_script(db_name, pred_raw_sql, gold):
            correct += 1
        else:
            wrong += 1
        total += 1

    print(correct / total)
