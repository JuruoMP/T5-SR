import os
import json
import copy
from nltk import word_tokenize


WHERE_OPS = [x.lower() for x in (
    "NOT",
    "BETWEEN",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "IN",
    "LIKE",
    "IS",
    "EXISTS",
)]
UNIT_OPS = ["none", "-", "+", "*", "/"]
AGG_OPS = [x.lower() for x in ("none", "MAX", "MIN", "COUNT", "SUM", "AVG")]
TABLE_TYPE = {
    "sql": "sql",
    "table_unit": "table_unit",
}

COND_OPS = [x.lower() for x in ("AND", "OR")]
SQL_OPS = [x.lower() for x in ("INTERSECT", "UNION", "EXCEPT")]
ORDER_OPS = [x.lower() for x in ("DESC", "ASC")]


class RawSqlDictParser:
    def __init__(self, table_path, lower_column=False):
        tables = json.load(open(table_path))
        self.tables = {x['db_id']: x for x in tables}
        self.lower_column = lower_column
        self.current_table = None
        self.table_mapping = None

    def unparse(self, db_id, sql_dict):
        self.current_table = self.tables[db_id]
        self.table_mapping = {}
        self.collect_table_mapping(sql_dict)
        assert len(self.table_mapping) > 0
        sql = self._unparse(sql_dict)[1:-1]
        sql = ' '.join(sql.split()).strip()
        return sql

    def _unparse(self, sql_dict):
        from_clause = self._unparse_from(sql_dict['from'], include_join=True)
        select_clause = self._unparse_select(sql_dict['select'])
        where_clause = groupby_clause = having_clause = orderby_clause = limit_clause = ''
        intersect_clause = except_clause = union_clause = ''
        if sql_dict['where']:
            where_clause = self._unparse_where(sql_dict['where'])
        if sql_dict['groupBy']:
            groupby_clause = self._unparse_groupby(sql_dict['groupBy'])
        if sql_dict['having']:
            having_clause = self._unparse_having(sql_dict['having'])
        if sql_dict['orderBy']:
            orderby_clause = self._unparse_orderby(sql_dict['orderBy'])
        if sql_dict['limit']:
            limit_clause = self._unparse_limit(sql_dict['limit'])
        if sql_dict['intersect']:
            intersect_clause = 'INTERSECT ' + self._unparse(sql_dict['intersect']).strip('(').strip(')')
        if sql_dict['except']:
            except_clause = 'EXCEPT ' + self._unparse(sql_dict['except']).strip('(').strip(')')
        if sql_dict['union']:
            union_clause = 'UNION ' + self._unparse(sql_dict['union']).strip('(').strip(')')
        sql = ' '.join([x for x in [select_clause, from_clause, where_clause,
                                    groupby_clause, having_clause, orderby_clause, limit_clause,
                                    intersect_clause, except_clause, union_clause] if x != ''])
        return '( ' + sql + ' )'

    def _unparse_select(self, _sel):
        is_distinct = _sel[0]
        sel_list = []
        for sel_item in _sel[1]:
            agg_id, val_unit = sel_item
            unit_op, col_unit1, col_unit2 = val_unit
            sel_item_str = self._unparse_col_unit(col_unit1)
            if unit_op != 0:
                # print('Warning: calculation between columns are used')
                sel_item2_str = self._unparse_col_unit(col_unit2)
                sel_item_str = ' '.join([sel_item_str, UNIT_OPS[unit_op], sel_item2_str])
            if agg_id > 0:
                sel_item_str = f'{AGG_OPS[agg_id]} ( {sel_item_str} )'
            sel_list.append(sel_item_str)
        sel_str = ', '.join(sel_list)
        if is_distinct:
            sel_str = 'DISTINCT ' + sel_str
        return 'SELECT ' + sel_str

    def _unparse_from(self, _from, include_join=True):
        table_units = _from['table_units']
        conds = _from['conds']
        table_unit_str_list = []
        for table_unit in table_units:
            table_type, table_id_or_sql = table_unit
            if table_type == 'table_unit':
                table_name = self.current_table['table_names_original'][table_id_or_sql]
                table_unit_str_list.append(table_name if not self.lower_column else table_name.lower())
            else:
                table_unit_str_list.append(self._unparse(table_id_or_sql))
        cond_str_list = self._unparse_condition(conds, return_list=True)
        assert all(x != 'or' for x in cond_str_list)
        cond_str_list = [x for x in cond_str_list if x not in ('and', 'or')]
        # assert len(table_unit_str_list) == len(cond_str_list) + 1  # assertion on number of join condition

        str_segs = []
        if include_join:
            if cond_str_list:
                # assert table_mapping is not None  # mapping is required for more than 2 tables
                # warning: this is a bug caused by sql dict
                # sql dict cannot distinguish columns from a single table with different table alias
                cond_str_list = self._unparse_condition(conds, return_list=True)
                assert all(x != 'or' for x in cond_str_list)
                cond_str_list = [x for x in cond_str_list if x not in ('and', 'or')]
            appear_table_set = set()
            str_segs.append(table_unit_str_list[0])
            if self.table_mapping and table_unit_str_list[0][0] != '(':
                table_map_id = self.table_mapping[self.current_table["table_names_original"].index(table_unit_str_list[0])]
                str_segs.append(f'AS T{table_map_id}')
                appear_table_set.add(table_map_id)
            condition_pairs = {}
            for cond_str in cond_str_list:
                col1_str, op_str, col2_str = cond_str.split()
                (t1, c1), (t2, c2) = col1_str.split('.'), col2_str.split('.')
                assert t1[0] == t2[0] == 'T'
                condition_pairs[(int(t1[1:]), int(t2[1:]))] = (c1, op_str, c2)
            for i in range(1, len(table_unit_str_list)):
                str_segs.append('JOIN')
                str_segs.append(table_unit_str_list[i])
                if self.table_mapping and table_unit_str_list[i][0] != '(':
                    table_map_id = self.table_mapping[self.current_table["table_names_original"].index(table_unit_str_list[i])]
                    str_segs.append(f'AS T{table_map_id}')
                    appear_table_set.add(table_map_id)
                    join_str_list = []
                    for other_table_map_id in appear_table_set:
                        if (table_map_id, other_table_map_id) in condition_pairs:
                            c1, op_str, c2 = condition_pairs[(table_map_id, other_table_map_id)]
                            join_str_list += [f'T{table_map_id}.{c1} {op_str} T{other_table_map_id}.{c2}', 'AND']
                        if (other_table_map_id, table_map_id) in condition_pairs:
                            c1, op_str, c2 = condition_pairs[(other_table_map_id, table_map_id)]
                            join_str_list += [f'T{other_table_map_id}.{c1} {op_str} T{table_map_id}.{c2}', 'AND']
                    if join_str_list:
                        join_str_list = ['ON'] + join_str_list[:-1]
                        str_segs += join_str_list
                    appear_table_set.add(table_map_id)
        else:
            str_segs.append(table_unit_str_list[0])
            for table_name in table_unit_str_list[1:]:
                str_segs += [',', table_name]
        
        return 'FROM ' + ' '.join(str_segs)

    def _unparse_where(self, _where):
        clause = 'WHERE ' + self._unparse_condition(_where)
        return clause

    def _unparse_groupby(self, _groupby):
        gb_str_list = []
        for gb_item in _groupby:
            gb_str = self._unparse_col_unit(gb_item)
            gb_str_list.append(gb_str)
        return 'GROUP BY ' + ', '.join(gb_str_list)

    def _unparse_orderby(self, _orderby):
        order_op_str = _orderby[0].upper()
        val_unit_str_list = []
        for val_unit in _orderby[1]:
            unit_op, col_unit1, col_unit2 = val_unit
            col_unit_str = self._unparse_col_unit(col_unit1)
            if unit_op != 0:
                # print('Warning: calculation between columns are used')
                col_unit2_str = self._unparse_col_unit(col_unit2)
                col_unit_str = ' '.join([col_unit_str, UNIT_OPS[unit_op], col_unit2_str])
            val_unit_str_list.append(col_unit_str)
        clause = 'ORDER BY ' + ', '.join(val_unit_str_list) + ' ' + order_op_str
        return clause

    def _unparse_having(self, _having):
        clause = 'HAVING ' + self._unparse_condition(_having)
        return clause

    def _unparse_limit(self, limit):
        return 'LIMIT ' + str(limit)

    def _unparse_col_unit(self, col_unit):
        agg_id, col_id, is_distinct = col_unit
        clause = ''
        table_id, column_name = self.current_table['column_names_original'][col_id]
        if table_id >= 0:
            # column_name = self.current_table['table_names_original'][table_id] + '.' + column_name
            if self.table_mapping:
                column_name = f'T{self.table_mapping[table_id]}' + '.' + column_name
            else:
                column_name = f'{self.current_table["table_names_original"][table_id]}.{column_name}'
        clause += column_name.lower() if self.lower_column else column_name
        if agg_id > 0:
            clause = AGG_OPS[agg_id] + ' ( ' + clause + ' ) '
        if is_distinct:
            clause = 'DISTINCT ' + clause
        return clause

    def _unparse_condition(self, condition, return_list=False):
        cond_str_list = []
        for cond_unit in condition:
            if cond_unit in ('and', 'or'):
                cond_str_list.append(cond_unit)
            else:
                # cond unit
                not_op, op_id, val_unit, val1, val2 = cond_unit
                op_str = WHERE_OPS[op_id]
                # val_unit
                unit_op, col_unit1, col_unit2 = val_unit
                col_unit_str = self._unparse_col_unit(col_unit1)
                if unit_op != 0:
                    # print('Warning: calculation between columns are used')
                    unit_op_str = UNIT_OPS[unit_op]
                    col_unit2_str = self._unparse_col_unit(col_unit2)
                    col_unit_str = ' '.join([col_unit_str, unit_op_str, col_unit2_str])
                val1_str = self._unparse_val(val1)
                val2_str = self._unparse_val(val2)
                if not_op:
                    assert op_str.lower() in ('in', 'like'), f"{op_str} found"  # todo: check here
                    op_str = 'NOT ' + op_str
                if 'between' not in op_str.lower():
                    cond_str_list.append(f'{col_unit_str} {op_str} {val1_str}')
                else:
                    assert op_str.lower() == 'between'
                    cond_str_list.append(f'{col_unit_str} {op_str} {val1_str} AND {val2_str}')
        if return_list is False:
            return ' '.join(cond_str_list)
        else:
            return cond_str_list

    def _unparse_val(self, val):
        if val is None:
            return None
        if isinstance(val, str):
            val_str = val
        elif isinstance(val, dict):
            val_str = self._unparse(val)
        elif isinstance(val, int) or isinstance(val, float):
            try:
                val = int(val)
            except:
                val = float(val)
            val_str = str(val)
            # val_str = 'value'
        else:
            val_str = self._unparse_col_unit(val)
        return val_str

    def collect_table_mapping(self, sql_dict):
        _from = sql_dict['from']
        table_units = _from['table_units']
        for table_unit in table_units:
            table_type, table_id_or_sql = table_unit
            if table_type == 'table_unit':
                if table_id_or_sql not in self.table_mapping:
                    self.table_mapping[table_id_or_sql] = len(self.table_mapping) + 1
                # todo: error when single table with multiple alias
        if sql_dict['from']['table_units'][0][0] == 'sql':
            self.collect_table_mapping(sql_dict['from']['table_units'][0][1])
        if sql_dict['where']:
            for i in range(len(sql_dict['where'])):
                if isinstance(sql_dict['where'][i], list) and isinstance(sql_dict['where'][i][3], dict):
                    self.collect_table_mapping(sql_dict['where'][i][3])
        for key in ('except', 'union', 'intersect'):
            if key in sql_dict and sql_dict[key]:
                self.collect_table_mapping(sql_dict[key])


if __name__ == '__main__':
    sql_parser = RawSqlDictParser('data/sparc/tables.json')
    dev_data = json.load(open('data/sparc/train.json'))
    bad_case, raw_bad_case = 0, 0
    correct, wrong = 0, 0

    for i in range(len(dev_data)):
        db_name = dev_data[i]['database_id']
        for turn in dev_data[i]['interaction']:
            sql_dict = turn['sql']
            raw_sql = turn['query']
            try:
                parsed_sql = sql_parser.unparse(db_name, sql_dict)
                print(parsed_sql)
            except:
                bad_case += 1
        print()

    print(f'#Case:{len(dev_data)}')
    print(f'Bad case: {bad_case}')
    print(f'Raw bad case: {raw_bad_case}')
    print(f'Correct: {correct}')
    print(f'Wrong: {wrong}')
