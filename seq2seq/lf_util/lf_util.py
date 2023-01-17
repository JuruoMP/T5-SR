import os
import json

from seq2seq.third_party.test_suite.process_sql import get_sql
from seq2seq.lf_util.sql2sem import Parser as SQL2SemParser
from seq2seq.lf_util.sem2sql import transform as semd2sql_transform


class LFUtil:
    def __init__(self, data_path, db_path=None):
        table_path = os.path.join(data_path, 'tables.json')
        tables = json.load(open(table_path))
        self.tables = {table['db_id']: table for table in tables}
        if db_path is None:
            db_path = os.path.join(data_path, 'database')
        self.db_path = db_path

        self.sql2sem_parser = SQL2SemParser()

    def sql_to_sqld(self, db_id, sql):
        db_table = self.tables[db_id]
        sqld = get_sql(db_table, sql)
        return sqld

    def sqld_to_semd(self, db_id, sql_d):
        db_table = self.tables[db_id]
        semql_d = self.sql2sem_parser.full_parse_wrapper({'sql': sql_d}, db_table)
        return semql_d

    def semd_to_sql(self, db_id, semd):
        db_table = self.tables[db_id]
        sql = semd2sql_transform([semd], db_table)
        return sql

    def semd_to_semql(self, db_id, semd):
        db_table = self.tables[db_id]
        db_table_view = self.tables_to_view(db_table)

        return sem2str(semd, db_table)

    def semql_to_semd(self, db_id, semql):
        db_table = self.tables[db_id]
        db_table_view = self.tables_to_view(db_table)
        sem_dict = get_sql(db_table_view, semql)
        return sem_dict

    @staticmethod
    def tables_to_view(table_schema):
        return table_schema


class SemLFUtil:
    def __init__(self, data_path, db_path=None):
        table_path = os.path.join(data_path, 'tables.json')
        tables = json.load(open(table_path))
        self.tables = {table['db_id']: table for table in tables}
        if db_path is None:
            db_path = os.path.join(data_path, 'database')
        self.db_path = db_path

    def sql_to_sqld(self, db_id, sql):
        db_table = self.tables[db_id]
        sqld = get_sql(db_table, sql)
        return sqld

    def sqld_to_sql(self, db_id, sqld):
        raise NotImplementedError

    def semql_to_sql(self, db_id, semql):
        db_table = self.tables[db_id]
        db_table_view = self.tables_to_view(db_table)
        semd = get_sql(db_table_view, semql)
        def semd_to_sqld(semd):
            raise NotImplementedError
        sqld = semd_to_sqld(semd)
        return sqld

    @staticmethod
    def tables_to_view(table_schema):
        raise NotImplementedError


if __name__ == '__main__':
    util = SemLFUtil('', 'data/database')