# Adapted from
# https://github.com/taoyds/spider/blob/master/process_sql.py

import os
import json
import spacy
from nltk.stem import WordNetLemmatizer
from seq2seq.lf_util.natsql.natsql_parser import NatsqlParser

nlp_model = spacy.load("en_core_web_lg")
lemmatizer = WordNetLemmatizer()


def calc_similarity(name1, name2):
    name1, name2 = name1.lower(), name2.lower()
    vec_sim = nlp_model(name1).similarity(nlp_model(name2))
    tok1, tok2 = name1.strip().split(), name2.strip().split()
    tok_match = len(set(tok1) & set(tok2))
    tok1 = [lemmatizer.lemmatize(x) for x in name1.strip().split('_')]
    tok2 = [lemmatizer.lemmatize(x) for x in name2.strip().split('_')]
    tok_lemma_match = len(set(tok1) & set(tok2))
    return tok_match + tok_lemma_match + vec_sim


class NatsqlFixParser(NatsqlParser):
    def __init__(self, table_path, db_dir):
        super().__init__(table_path, db_dir)

    def parse_table_unit(self, toks, start_idx, tables_with_alias, schema):
        """
            :returns next idx, table id, table name
        """
        idx = start_idx
        len_ = len(toks)
        if toks[idx] in tables_with_alias:
            key = tables_with_alias[toks[idx]]
        else:
            candidate_tables = [(table, self.calc_similarity(toks[idx], table)) for table in tables_with_alias]
            best_candidate = sorted(candidate_tables, key=lambda x: x[1], reverse=True)[0]
            key = best_candidate[0]

        if idx + 1 < len_ and toks[idx + 1] == "as":
            idx += 3
        else:
            idx += 1

        return idx, schema.idMap[key], key

    def parse_col(self, toks, start_idx, tables_with_alias, schema, default_tables=None):
        """
            :returns next idx, column id
        """
        tok = toks[start_idx]
        if tok == "*":
            assert False, "Should not achieve here"
            return start_idx + 1, schema.idMap[tok]

        if tok == "@.@":
            return start_idx + 1, "@.@"

        if '.' in tok:  # if token is a composite
            alias, col = tok.split('.')
            if alias not in tables_with_alias:
                candidate_tables = [(table, self.calc_similarity(alias, table)) for table in tables_with_alias]
                best_candidate = sorted(candidate_tables, key=lambda x: x[1], reverse=True)[0]
                alias = best_candidate[0]
            key = tables_with_alias[alias] + "." + col
            if key not in schema.idMap and toks[start_idx + 1] == "(":
                new_key = key
                for i in range(start_idx + 1, len(toks)):
                    new_key += toks[i]
                    if new_key in schema.idMap:
                        return i + 1, schema.idMap[new_key]
            if key not in schema.idMap:
                candidate_keys = [(k, self.calc_similarity(k, key)) for k in schema.idMap]
                best_candidate = sorted(candidate_keys, key=lambda x: x[1], reverse=True)[0]
                key = best_candidate[0]
            return start_idx + 1, schema.idMap[key]

        assert default_tables is not None and len(default_tables) > 0, "Default tables should not be None or empty"

        assert False, "Should not achieve here"
        for alias in default_tables:
            table = tables_with_alias[alias]
            if tok in schema.schema[table]:
                key = table + "." + tok
                if key not in schema.idMap and toks[start_idx + 1] == "(":
                    new_key = key
                    for i in range(start_idx + 1, len(toks)):
                        new_key += toks[i]
                        if new_key in schema.idMap:
                            return i + 1, schema.idMap[new_key]

                return start_idx + 1, schema.idMap[key]

        assert False, "Error col: {}".format(tok)

    @staticmethod
    def calc_similarity(name1, name2):
        name1, name2 = name1.lower(), name2.lower()
        vec_sim = nlp_model(name1).similarity(nlp_model(name2))
        tok1, tok2 = name1.strip().split(), name2.strip().split()
        tok_match = len(set(tok1) & set(tok2))
        tok1 = [lemmatizer.lemmatize(x) for x in name1.strip().split('_')]
        tok2 = [lemmatizer.lemmatize(x) for x in name2.strip().split('_')]
        tok_lemma_match = len(set(tok1) & set(tok2))
        return tok_match + tok_lemma_match + vec_sim


if __name__ == '__main__':
    natsql_parser = NatsqlFixParser('data/spider_natsql/tables_for_natsql.json', 'data/database')
    natsql_data = json.load(open('data/spider_natsql/dev.json'))
    n_correct, n_total = 0, 0
    for example in natsql_data:
        db_id = example['db_id']
        natsql = example['NatSQL']
        sql = natsql_parser.natsql_to_sql(db_id, natsql)
        if natsql_parser.check_equal_script(db_id, sql, example['query']) is True:
            n_correct += 1
        n_total += 1
    print(f'{n_correct} / {n_total} = {n_correct / n_total * 100:.2f}')
