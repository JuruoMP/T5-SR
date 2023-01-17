import os
import json
from nltk import word_tokenize

from seq2seq.lf_util.tree_algorithm import dijkstra, steiner_tree, check_connectitvity
from seq2seq.lf_util.utils import tuple_to_list, list_to_set
from seq2seq.lf_util.ssql.process_sql import get_sql, get_schema, Schema
from seq2seq.lf_util.raw_sql_dict_parser import RawSqlDictParser
from seq2seq.lf_util.evaluation import get_evaluator, evaluate_pair, evaluate_pair_exec, build_foreign_key_map_from_json

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


class SqlDictParser:
    def __init__(self, table_path, lower_column=False):
        tables = json.load(open(table_path))
        self.tables = {x['db_id']: x for x in tables}
        self.current_table = None
        self.lower_column = lower_column

    def unparse(self, db_id, sql_dict):
        self.current_table = self.tables[db_id]
        sql = self._unparse(sql_dict)[1:-1]
        sql = ' '.join(sql.split()).strip()
        return sql

    def _unparse(self, sql_dict):
        select_clause = self._unparse_select(sql_dict['select'])
        from_clause = self._unparse_from(sql_dict['from'], include_join=False)
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
            intersect_clause = 'intersect ' + self._unparse(sql_dict['intersect']).strip('(').strip(')')
        if sql_dict['except']:
            except_clause = 'except ' + self._unparse(sql_dict['except']).strip('(').strip(')')
        if sql_dict['union']:
            union_clause = 'union ' + self._unparse(sql_dict['union']).strip('(').strip(')')
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
            sel_str = 'distinct ' + sel_str
        return 'select ' + sel_str

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
            table_unit_str_list.append(',')
        cond_str_list = self._unparse_condition(conds, return_list=True)
        assert all(x != 'or' for x in cond_str_list)
        cond_str_list = [x for x in cond_str_list if x not in ('and', 'or')]
        # assert len(table_unit_str_list) == len(cond_str_list) + 1  # assertion on number of join condition
        str_segs = table_unit_str_list[:-1]
        if include_join:
            for i in range(1, len(table_unit_str_list)):
                str_segs.append('JOIN')
                str_segs.append(table_unit_str_list[i])
            if cond_str_list:
                str_segs.append('ON')
                str_segs.append(cond_str_list[0])
                for i in range(1, len(cond_str_list)):
                    str_segs.append('AND')
                    str_segs.append(cond_str_list[i])
        return 'from ' + ' '.join(str_segs)

    def _unparse_where(self, _where):
        clause = 'where ' + self._unparse_condition(_where)
        return clause

    def _unparse_groupby(self, _groupby):
        gb_str_list = []
        for gb_item in _groupby:
            gb_str = self._unparse_col_unit(gb_item)
            gb_str_list.append(gb_str)
        return 'group by ' + ', '.join(gb_str_list)

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
        clause = 'order by ' + ', '.join(val_unit_str_list) + ' ' + order_op_str
        return clause

    def _unparse_having(self, _having):
        clause = 'having ' + self._unparse_condition(_having)
        return clause

    def _unparse_limit(self, limit):
        return 'limit ' + str(limit)

    def _unparse_col_unit(self, col_unit):
        agg_id, col_id, is_distinct = col_unit
        clause = ''
        table_id, column_name = self.current_table['column_names_original'][col_id]
        if table_id >= 0:
            column_name = self.current_table['table_names_original'][table_id] + '.' + column_name
        clause += column_name.lower() if self.lower_column else column_name
        if agg_id > 0:
            clause = AGG_OPS[agg_id] + ' ( ' + clause + ' ) '
        if is_distinct:
            clause = 'distinct ' + clause
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
                    op_str = 'not ' + op_str
                if 'between' not in op_str.lower():
                    cond_str_list.append(f'{col_unit_str} {op_str} {val1_str}')
                else:
                    assert op_str.lower() == 'between'
                    cond_str_list.append(f'{col_unit_str} {op_str} {val1_str} and {val2_str}')
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
        else:
            val_str = self._unparse_col_unit(val)
        return val_str


class SqlStrParser:
    CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'having', 'order', 'limit', 'intersect', 'union', 'except')

    def __init__(self, table_path):
        tables = json.load(open(table_path))
        self.tables = {x['db_id']: x for x in tables}
        self.current_table = None
        self.clause_list = ['select', 'from', 'where', 'groupby', 'having', 'orderby', 'limit', 'intersect', 'except',
                            'union']
        self.gold = None

    def parse(self, db_id, sql):
        self.current_table = self.tables[db_id]
        tokens = self._tokenize_sql(sql)
        sql_dict, _ = self._parse_sql(tokens, 0)
        sql_dict = self._heuristic_from(sql_dict)
        return sql_dict

    def _parse_sql(self, tokens, start_idx):
        sql_dict = {}
        idx = start_idx
        is_block = False
        if tokens[start_idx] == '(':
            is_block = True
            idx += 1
        if 'select' in self.clause_list:
            sql_dict['select'], idx = self._parse_select(tokens, idx)
        if 'from' in self.clause_list:
            sql_dict['from'], idx = self._parse_from(tokens, idx)
        if 'where' in self.clause_list:
            sql_dict['where'], idx = self._parse_where(tokens, idx)
        if 'groupby' in self.clause_list:
            sql_dict['groupBy'], idx = self._parse_groupby(tokens, idx)
        if 'having' in self.clause_list:
            sql_dict['having'], idx = self._parse_having(tokens, idx)
        if 'orderby' in self.clause_list:
            sql_dict['orderBy'], idx = self._parse_orderby(tokens, idx)
        if 'limit' in self.clause_list:
            sql_dict['limit'], idx = self._parse_limit(tokens, idx)
        if 'intersect' in self.clause_list:
            sql_dict['intersect'], idx = self._parse_uie(tokens, idx, 'intersect')
        if 'union' in self.clause_list:
            sql_dict['union'], idx = self._parse_uie(tokens, idx, 'union')
        if 'except' in self.clause_list:
            sql_dict['except'], idx = self._parse_uie(tokens, idx, 'except')
        if is_block:
            idx += 1
        return sql_dict, idx

    def _parse_select(self, tokens, start_idx):
        #print("tokens")
        #print(tokens)
        #print(start_idx)
        assert tokens[start_idx] == 'select', 'error parsing select'
        idx = start_idx + 1
        if tokens[idx] == 'distinct':
            is_distinct = True
            idx += 1
        else:
            is_distinct = False

        val_unit_list = []
        while idx < len(tokens) and tokens[idx] not in self.CLAUSE_KEYWORDS:
            agg_id = AGG_OPS.index('none')
            if tokens[idx] in AGG_OPS:
                agg_id = AGG_OPS.index(tokens[idx])
                idx += 1
            val_unit, idx = self._parse_val_unit(tokens, idx)
            val_unit_list.append([agg_id, val_unit])
            if idx < len(tokens) and tokens[idx] == ',':
                idx += 1
        return [is_distinct, val_unit_list], idx

    def _parse_from(self, tokens, start_idx):
        assert tokens[start_idx] == 'from', 'error parsing from'
        idx = start_idx + 1

        default_tables = []
        table_units = []
        conds = []

        while idx < len(tokens):
            is_block = False
            if tokens[idx] == '(':
                is_block = True
                idx += 1

            if tokens[idx] == 'select':
                sql, idx = self._parse_sql(tokens, idx)
                table_units.append([TABLE_TYPE['sql'], sql])
            else:
                while idx < len(tokens) and tokens[idx] not in (')', ';') + self.CLAUSE_KEYWORDS:
                    table_name = tokens[idx]
                    idx += 1
                    table_unit = self._parse_table(table_name)
                    table_units.append([TABLE_TYPE['table_unit'], table_unit])
                    default_tables.append(table_name)
                    if idx < len(tokens) and tokens[idx] == ',':
                        idx += 1
                    else:
                        break

            if is_block:
                assert tokens[idx] == ')'
                idx += 1
            if idx < len(tokens) and tokens[idx] in self.CLAUSE_KEYWORDS + (")", ";"):
                break

        return {'table_units': table_units, 'conds': conds}, idx

    def _parse_where(self, tokens, start_idx):
        if start_idx >= len(tokens) or tokens[start_idx] != 'where':
            return [], start_idx
        idx = start_idx + 1
        conds, idx = self._parse_condition(tokens, idx)
        return conds, idx

    def _parse_groupby(self, tokens, start_idx):
        if start_idx >= len(tokens) or tokens[start_idx] != 'group' or tokens[start_idx + 1] != 'by':
            return [], start_idx
        idx = start_idx + 2
        col_unit_list = []
        while idx < len(tokens) and tokens[idx] not in (')', ';') + self.CLAUSE_KEYWORDS:
            col_unit, idx = self._parse_col_unit(tokens, idx)
            col_unit_list.append(col_unit)
            if idx < len(tokens) and tokens[idx] == ',':
                idx += 1
            else:
                break
        return col_unit_list, idx

    def _parse_having(self, tokens, start_idx):
        if start_idx >= len(tokens) or tokens[start_idx] != 'having':
            return [], start_idx
        idx = start_idx + 1
        conds, idx = self._parse_condition(tokens, idx)
        return conds, idx

    def _parse_orderby(self, tokens, start_idx):
        if start_idx >= len(tokens) or tokens[start_idx] != 'order':
            return [], start_idx
        assert tokens[start_idx] == 'order' and tokens[start_idx + 1] == 'by'
        idx = start_idx + 2
        val_unit_list = []
        order_type = 'asc'
        while idx < len(tokens) and tokens[idx] not in (')', ';') + self.CLAUSE_KEYWORDS:
            val_unit, idx = self._parse_val_unit(tokens, idx)
            val_unit_list.append(val_unit)
            if idx < len(tokens) and tokens[idx] in ORDER_OPS:
                order_type = tokens[idx]
                idx += 1
            if idx < len(tokens) and tokens[idx] == ',':
                idx += 1
            else:
                break
        return [order_type, val_unit_list], idx

    def _parse_limit(self, tokens, start_idx):
        if start_idx >= len(tokens) or tokens[start_idx] != 'limit':
            return None, start_idx
        idx = start_idx + 1
        n_limit = int(tokens[idx])
        return n_limit, idx + 1

    def _parse_uie(self, tokens, start_idx, type_str):
        if start_idx >= len(tokens) or tokens[start_idx] != type_str:
            return None, start_idx
        idx = start_idx + 1
        sql, idx = self._parse_sql(tokens, idx)
        return sql, idx

    def _parse_val_unit(self, tokens, start_idx):
        idx = start_idx
        is_block = False
        if tokens[idx] == '(':
            is_block = True
            idx += 1

        col_unit1, col_unit2, unit_op = None, None, UNIT_OPS.index('none')
        col_unit1, idx = self._parse_col_unit(tokens, idx)
        if idx < len(tokens) and tokens[idx] in UNIT_OPS:
            unit_op = UNIT_OPS.index(tokens[idx])
            idx += 1
            col_unit2, idx = self._parse_col_unit(tokens, idx)

        if is_block:
            idx += 1
        return [unit_op, col_unit1, col_unit2], idx

    def _parse_col_unit(self, tokens, start_idx):
        idx = start_idx
        is_block = False
        if tokens[idx] == '(':
            is_block = True
            idx += 1

        is_distinct = False
        if tokens[idx] == 'distinct':
            is_distinct = True
            idx += 1
        agg_id = AGG_OPS.index('none')
        if tokens[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(tokens[idx])
            idx += 1
        col_id, idx = self._parse_col(tokens, idx)

        if is_block:
            idx += 1

        return [agg_id, col_id, is_distinct], idx

    def _parse_table(self, table_name):
        table_unit = -100
        all_table_names = self.current_table['table_names_original']
        for table_id in range(len(all_table_names)):
            if all_table_names[table_id].lower() == table_name:
                table_unit = table_id
                break
        assert table_unit >= 0
        return table_unit

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
        idx += 1
        if is_block:
            idx += 1
        assert col_column_id != -100
        return col_column_id, idx

    def _parse_condition(self, tokens, start_idx):
        idx = start_idx
        conds = []
        if tokens[idx] in COND_OPS:
            conds.append(tokens[idx])
        while idx < len(tokens):
            val_unit, idx = self._parse_val_unit(tokens, idx)
            not_op = False
            if tokens[idx] == 'not':
                not_op = True
                idx += 1
            assert idx < len(tokens) and tokens[idx] in WHERE_OPS, "Error condition: idx: {}, tok: {}".format(idx,
                                                                                                              tokens[
                                                                                                                  idx])
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

    def _tokenize_sql(self, sql):
        sql = str(sql)
        sql = sql.replace('. ', '.').replace(' .', '.')
        sql = sql.replace("\'", "\"")  # ensures all string values wrapped by "" problem??
        quote_idxs = [idx for idx, char in enumerate(sql) if char == '"']
        assert len(quote_idxs) % 2 == 0, "Unexpected quote"

        # keep string value as token
        vals = {}
        for i in range(len(quote_idxs) - 1, -1, -2):
            qidx1 = quote_idxs[i - 1]
            qidx2 = quote_idxs[i]
            val = sql[qidx1: qidx2 + 1]
            key = "__val_{}_{}__".format(qidx1, qidx2)
            sql = sql[:qidx1] + key + sql[qidx2 + 1:]
            vals[key] = val

        toks = [word.lower() for word in word_tokenize(sql)]
        # replace with string value token
        for i in range(len(toks)):
            if toks[i] in vals:
                toks[i] = vals[toks[i]]

        # find if there exists !=, >=, <=
        eq_idxs = [idx for idx, tok in enumerate(toks) if  # todo: delete in and like
                   tok == "=" or tok == "in" or tok == "like"]  # make 'not in' and 'not like' together
        eq_idxs.reverse()
        prefix = ('!', '>', '<')
        for eq_idx in eq_idxs:
            pre_tok = toks[eq_idx - 1]
            if pre_tok in prefix:
                assert toks[eq_idx] == '='
                toks = toks[:eq_idx - 1] + [pre_tok + "="] + toks[eq_idx + 1:]
            # elif pre_tok == 'not':
            #    toks = toks[:eq_idx - 1] + [pre_tok + " " + toks[eq_idx]] + toks[
            #                                                                eq_idx + 1:]  # make 'not in' and 'not like' together
        return toks

    def _heuristic_from(self, sql_dict):
        db_schema = self.current_table
        _select, _from, _where = sql_dict['select'], sql_dict['from'], sql_dict['where']
        _groupby, _having, _orderby = sql_dict['groupBy'], sql_dict['having'], sql_dict['orderBy']
        all_table_ids = []

        if _from['table_units'][0][0] == 'sql':
            assert len(_from['table_units']) == 1
            _from['table_units'][0][1] = self._heuristic_from(_from['table_units'][0][1])

        else:
            # select
            _, sel_list = _select
            for agg_id, val_unit in sel_list:
                _, col_unit1, col_unit2 = val_unit
                if col_unit1 is not None:
                    all_table_ids.append(self.current_table['column_names'][col_unit1[1]][0])
                if col_unit2 is not None:
                    all_table_ids.append(self.current_table['column_names'][col_unit2[1]][0])
            # from
            table_units, conds = _from['table_units'], _from['conds']
            for table_unit in table_units:
                table_type, table_id = table_unit
                assert table_type == 'table_unit'
                all_table_ids.append(table_id)
            for cond_unit in conds:
                if cond_unit in ('and', 'or'):
                    continue
                _, _, val_unit, val1, val2 = cond_unit
                _, col_unit1, col_unit2 = val_unit
                if col_unit1 is not None:
                    all_table_ids.append(self.current_table['column_names'][col_unit1[1]][0])
                if col_unit2 is not None:
                    all_table_ids.append(self.current_table['column_names'][col_unit2[1]][0])
                if isinstance(val1, dict):
                    val1 = self._heuristic_from(val1)
                    cond_unit[3] = val1
                if isinstance(val2, dict):
                    val2 = self._heuristic_from(val2)
                    cond_unit[4] = val2
            # where
            for cond_unit in _where:
                if cond_unit in ('and', 'or'):
                    continue
                _, _, val_unit, val1, val2 = cond_unit
                _, col_unit1, col_unit2 = val_unit
                if col_unit1 is not None:
                    all_table_ids.append(self.current_table['column_names'][col_unit1[1]][0])
                if col_unit2 is not None:
                    all_table_ids.append(self.current_table['column_names'][col_unit2[1]][0])
                if isinstance(val1, dict):
                    val1 = self._heuristic_from(val1)
                    cond_unit[3] = val1
                if isinstance(val2, dict):
                    val2 = self._heuristic_from(val2)
                    cond_unit[4] = val2
            # groupBy
            for col_unit in _groupby:
                all_table_ids.append(self.current_table['column_names'][col_unit[1]][0])
            # having
            if _having:
                for cond_unit in _having:
                    if cond_unit in ('and', 'or'):
                        continue
                    _, _, val_unit, val1, val2 = cond_unit
                    _, col_unit1, col_unit2 = val_unit
                    if col_unit1 is not None:
                        all_table_ids.append(self.current_table['column_names'][col_unit1[1]][0])
                    if col_unit2 is not None:
                        all_table_ids.append(self.current_table['column_names'][col_unit2[1]][0])
                    if isinstance(val1, dict):
                        val1 = self._heuristic_from(val1)
                        cond_unit[3] = val1
                    if isinstance(val2, dict):
                        val2 = self._heuristic_from(val2)
                        cond_unit[4] = val2
            # orderBy
            if _orderby:
                for val_unit in _orderby[1]:
                    col_unit = val_unit[1]
                    all_table_ids.append(self.current_table['column_names'][col_unit[1]][0])

            # collect appear tables
            n_nodes = len(db_schema['table_names'])
            appear_table_set = set(all_table_ids) - {-1}
            # print(f'Appear table set: {appear_table_set}')

            # build graph and find MST
            edges, heuristic_edges = self._build_edges(db_schema)
            if False:
                connection_edges = self._minimum_spanning_tree(list(range(n_nodes)), list(appear_table_set), edges)
            else:
                connection_edges = self._minimum_spanning_tree(list(range(n_nodes)), list(appear_table_set), heuristic_edges)
            # print(f'Edges: {connection_edges}')

            # reformat to ``from'' dict
            from_table_list = list(appear_table_set)
            from_condition_list = []
            for (table_id1, col_id1), (table_id2, col_id2) in connection_edges:
                # join table_id1 with table_id2 on table_id1.col_id1 = table_id2.col_id2
                from_table_list += [table_id1, table_id2]
                cond_unit = (False, WHERE_OPS.index('='),
                             [UNIT_OPS.index('none'), [AGG_OPS.index('none'), col_id1, False], None],
                             [AGG_OPS.index('none'), col_id2, False],
                             None)
                from_condition_list += [cond_unit, 'and']
            sql_dict['from'] = {
                'table_units': [['table_unit', table_id] for table_id in set(from_table_list)],
                'conds': tuple_to_list(from_condition_list[:-1])  # strip last ``and''
            }

        for key in ('union', 'except', 'intersect'):
            if sql_dict[key]:
                sql_dict[key] = self._heuristic_from(sql_dict[key])

        return sql_dict

    @staticmethod
    def _build_edges(db_schema):
        column_to_table = {}
        for column_id in range(1, len(db_schema['column_names_original'])):
            table_id, column_name = db_schema['column_names_original'][column_id]
            column_to_table[column_id] = table_id
        edges = []
        for col_id1, col_id2 in db_schema['foreign_keys']:
            table_id1, table_id2 = column_to_table[col_id1], column_to_table[col_id2]
            weight = 0.8
            edges.append(((table_id1, col_id1), (table_id2, col_id2), weight))

        real_edges = [(st[0], ed[0]) for st, ed, weight in edges]
        heuristic_edges = edges[:]
        if check_connectitvity(len(column_to_table), real_edges) is False:
            column_details = []
            for column_id, (table_id, column_name) in enumerate(db_schema['column_names']):
                table_name = db_schema['table_names'][table_id]
                full_column_name = table_name + ' ' + column_name
                column_details.append((column_id, column_name, full_column_name))
            for i in range(len(column_details)):
                for j in range(i + 1, len(column_details)):
                    col_detail1, col_detail2 = column_details[i], column_details[j]
                    if col_detail1[1] == col_detail2[1] or \
                            col_detail1[2] in col_detail2[2] or col_detail2[2] in col_detail1[2]:
                        weight = 1 if 'id' in col_detail1[1] else 1.5
                        if col_detail1[0] != 0 and col_detail2[0] != 0:
                            heuristic_edges.append(((column_to_table[col_detail1[0]], col_detail1[0]),
                                                    (column_to_table[col_detail2[0]], col_detail2[0]), weight))
        return edges, heuristic_edges

    @staticmethod
    def _minimum_spanning_tree(all_nodes, used_nodes, edges):
        if len(used_nodes) == 1:
            return []
        else:
            adj_mat = [[0xffff for _ in range(len(all_nodes))] for _ in range(len(all_nodes))]
            for i in range(len(all_nodes)):
                adj_mat[i][i] = 0
            info_dict = {}
            for (node1, info1), (node2, info2), weight in edges:
                if weight < adj_mat[node1][node2]:
                    adj_mat[node1][node2] = adj_mat[node2][node1] = weight
                    info_dict[(node1, node2)] = (info1, info2)
                    info_dict[(node2, node1)] = (info2, info1)
            if len(used_nodes) == 2:
                selected_edges = dijkstra(adj_mat, src=used_nodes[0], tgt=used_nodes[1])
            else:
                selected_edges = steiner_tree(adj_mat, used_nodes)
                if selected_edges is None:
                    selected_edges = []

            ret = []
            for edge in selected_edges:
                node1, node2 = edge
                info1, info2 = info_dict[edge]
                ret.append(((node1, info1), (node2, info2)))
            return ret


class SqlParser:
    def __init__(self, table_path, db_dir='data/database'):
        self.table_path = table_path
        self.db_dir = db_dir
        self.db_paths = {}
        self.schemas = {}
        table_maps = {x['db_id']: x for x in json.load(open(table_path), encoding='utf-8')}
        miss_db_list = []
        for db_name in table_maps.keys():
            db_path = os.path.join(db_dir, db_name, db_name + ".sqlite")
            if os.path.exists(db_path):
                self.db_paths[db_name] = db_path
                table_info = table_maps[db_name]
                table_info.update({
                    'table_names_original': [x.lower() for x in table_info['table_names_original']],
                    'column_names_original': [[x[0], x[1].lower()] for x in table_info['column_names_original']]
                })
                self.schemas[db_name] = (table_info, Schema(get_schema(db_path)))
            else:
                raise Exception(f'DB file {db_path} not found.')
                miss_db_list.append(db_name)
        if miss_db_list:
            print(f'Missing database: {", ".join(miss_db_list)}')

        self.sql_dict_parser = SqlDictParser(table_path)
        self.sql_str_parser = SqlStrParser(table_path)
        self.raw_sql_dict_parser = RawSqlDictParser(table_path)
        self.script_evaluator = get_evaluator()
        self.kmaps = build_foreign_key_map_from_json(table_path)

    def sql_to_dict(self, db_id, sql):
        # convert simple_sql to simple_dict
        return self.sql_str_parser.parse(db_id, sql)

    def dict_to_sql(self, db_id, sql_dict):
        # convert simple_dict to simple_sql
        return self.sql_dict_parser.unparse(db_id, sql_dict)

    def check_equal(self, sql_d1, sql_d2):
        sql_d1, sql_d2 = list_to_set(tuple_to_list(sql_d1)), list_to_set(tuple_to_list(sql_d2))

        def swap_condition(sql_d):
            if sql_d and sql_d['from']:
                conds = sql_d['from']['conds']
                for i in range(len(conds)):
                    if conds[i] in ('and', 'or'):
                        continue
                    col1 = conds[i][2][1]
                    col2 = conds[i][3]
                    if str(col1) > str(col2):
                        conds[i][2][1] = col2
                        conds[i][3] = col1
                sql_d['from']['conds'] = sorted(conds, key=lambda x: str(x))
            for key in ('union', 'except', 'intersect'):
                if sql_d and sql_d[key]:
                    sql_d[key] = swap_condition(sql_d[key])
            return sql_d

        new_sql_d1, new_sql_d2 = swap_condition(sql_d1), swap_condition(sql_d2)

        # if new_sql_d1 != new_sql_d2:
        #     print('Unequal sql dict found.')

        return new_sql_d1 == new_sql_d2

    def check_equal_script(self, db_name, pred_sql, gold_sql):
        if pred_sql == '':
            return False
        return evaluate_pair(self.script_evaluator, self.kmaps, self.schemas[db_name][1], db_name, pred_sql, gold_sql)

    def check_exec_match_script(self, db_name, pred_sql, gold_sql, pred_dict, gold_dict):
        if pred_sql == '':
            return False
        return evaluate_pair_exec(self.script_evaluator, self.kmaps, self.schemas[db_name][1],
                                  os.path.join(self.db_dir, db_name, f'{db_name}.sqlite'),
                                  pred_sql, gold_sql, pred_dict, gold_dict)

    def check_equal_without_from(self, sql_d1, sql_d2):
        def del_from(d):
            if isinstance(d, dict):
                if 'from' in d:
                    del d['from']
                for k in d:
                    del_from(d[k])
            elif isinstance(d, list):
                d = [del_from(x) for x in d]
            return d

        sql_d1, sql_d2 = del_from(sql_d1), del_from(sql_d2)
        return sql_d1 == sql_d2

    def raw_sql_to_dict(self, db_id, sql, evaluate=False):
        # notice: returned dict is evaluate dict
        schema = self.schemas[db_id]
        sql = get_sql(schema[1], sql)
        if not evaluate:
            sql = self.dict_name_to_id(db_id, sql)
        return sql

    def dict_name_to_id(self, db_id, sql_dict):
        # convert column names into column_ids
        def col_name_to_id(col_unit, schema_dict):
            if col_unit is None:
                return None
            agg_id, col_name, is_distinct = col_unit
            col_name = col_name.strip('_')
            if col_name == 'all':
                col_id = 0
            else:
                col_id = -1
                tgt_table_name, tgt_column_name = col_name.split('.')
                for i in range(1, len(schema_dict['column_names_original'])):
                    table_id, column_name = schema_dict['column_names_original'][i]
                    table_name = schema_dict['table_names_original'][table_id]
                    if table_name == tgt_table_name and column_name == tgt_column_name:
                        col_id = i
                        break
                assert col_id != -1
            col_unit = (agg_id, col_id, is_distinct)
            return col_unit

        def table_name_to_id(table_unit, schema_dict):
            table_type, col_unit_or_sql = table_unit
            if table_type == 'table_unit':
                table_name = col_unit_or_sql.strip('_')
                table_id = schema_dict['table_names_original'].index(table_name)
                assert table_id != -1
                col_unit_or_sql = table_id
            else:
                col_unit_or_sql = name_to_id(col_unit_or_sql, schema_dict)
            return (table_type, col_unit_or_sql)

        def cond_to_id(condition, schema_dict):
            new_condition = []
            for cond_unit in condition:
                if isinstance(cond_unit, str):
                    new_condition.append(cond_unit)
                else:
                    not_op, op_id, val_unit, val1, val2 = cond_unit
                    val_unit = (val_unit[0],
                                col_name_to_id(val_unit[1], schema_dict),
                                col_name_to_id(val_unit[2], schema_dict))
                    convert_switch_fn = lambda x: col_name_to_id if isinstance(x, tuple) else \
                        (name_to_id if isinstance(x, dict) else (lambda x, y: x))
                    val1 = convert_switch_fn(val1)(val1, schema_dict)
                    val2 = convert_switch_fn(val2)(val2, schema_dict)
                    new_cond_unit = (not_op, op_id, val_unit, val1, val2)
                    new_condition.append(new_cond_unit)
            return new_condition

        def name_to_id(sql_dict, schema_dict):
            if sql_dict['select']:
                select_cols = sql_dict['select'][1]
                for i in range(len(select_cols)):
                    agg_id, (unit_op, col1, col2) = select_cols[i]
                    col1, col2 = col_name_to_id(col1, schema_dict), col_name_to_id(col2, schema_dict)
                    select_cols[i] = (agg_id, (unit_op, col1, col2))
            if sql_dict['from']:
                sql_dict['from']['table_units'] = [table_name_to_id(x, schema_dict) for x in
                                                   sql_dict['from']['table_units']]
                sql_dict['from']['conds'] = cond_to_id(sql_dict['from']['conds'], schema_dict)
            if sql_dict['where']:
                condition = sql_dict['where']
                sql_dict['where'] = cond_to_id(condition, schema_dict)
            if sql_dict['having']:
                condition = sql_dict['having']
                sql_dict['having'] = cond_to_id(condition, schema_dict)
            if sql_dict['groupBy']:
                sql_dict['groupBy'] = [col_name_to_id(x, schema_dict) for x in sql_dict['groupBy']]
            if sql_dict['orderBy']:
                val_unit_list = sql_dict['orderBy'][1]
                for i, val_unit in enumerate(val_unit_list):
                    new_val_unit = (val_unit[0],
                                    col_name_to_id(val_unit[1], schema_dict),
                                    col_name_to_id(val_unit[2], schema_dict))
                    val_unit_list[i] = new_val_unit
            for sub_clause in ('intersect', 'union', 'except'):
                if sql_dict[sub_clause]:
                    sql_dict[sub_clause] = name_to_id(sql_dict[sub_clause], schema_dict)
            return sql_dict

        schema = self.schemas[db_id]
        sql = name_to_id(sql_dict, schema[0])
        return sql

    def dict_to_raw_sql(self, db_id, sql_d):
        if sql_d == {}:
            return ''
        return self.raw_sql_dict_parser.unparse(db_id, sql_d)


if __name__ == '__main__':
    import os

    sql_parser = SqlParser('data/spider/tables.json', db_dir='data/database')
    dev_data = json.load(open('data/spider/train_spider.json'))
    bad_case, raw_bad_case = 0, 0
    correct, wrong = 0, 0
    for i in range(len(dev_data)):
    # for i in range(6961, 6962):
        # print(i)
        # print(dev_data[i]['question'])
        db_name = dev_data[i]['db_id']
        sql_dict = dev_data[i]['sql']
        raw_sql = dev_data[i]['query']

        # print(raw_sql)

        my_sql = sql_parser.dict_to_sql(db_name, sql_dict)
        # print(my_sql)

        # print()

        if True:
            my_sql_d = sql_parser.sql_to_dict(db_name, my_sql)
        else:
            my_sql_d = {}
            bad_case += 1
        # print(json.dumps(new_sql_d, indent=2))
        try:
            raw_sql_dict = sql_parser.raw_sql_to_dict(db_name, raw_sql)
        except:
            raw_sql_dict = {}
            raw_bad_case += 1

        # label = sql_parser.check_equal(raw_sql_dict, my_sql_d)
        my_raw_sql = sql_parser.dict_to_raw_sql(db_name, my_sql_d)
        try:
            label = sql_parser.check_equal_script(db_name, my_raw_sql, raw_sql)
        except:
            bad_case += 1
            label = False

        if label is True:
            correct += 1
        else:
            wrong += 1
            # print(i)
            # print(raw_sql)
            # print(my_sql)
            # print()

    print(f'#Case:{len(dev_data)}')
    print(f'Bad case: {bad_case}')
    print(f'Raw bad case: {raw_bad_case}')
    print(f'Correct: {correct}')
    print(f'Wrong: {wrong}')


# if __name__ == '__main__':
#     import os
#     from tqdm import trange
#
#     sql_parser = SqlParser('data/spider/tables.json', db_dir='data/spider/database')
#     pred_result = open('seq2seq/tmp/pred.txt', encoding='utf-8').readlines()
#     correct, wrong = 0, 0
#     error_convert, error_check = 0, 0
#     for i in range(len(pred_result)):
#         # print(dev_data[i]['question'])
#         db_name, pred_sql, raw_sql = pred_result[i].strip().split('\t')
#         try:#if True:
#             pred_sql_dict = sql_parser.sql_to_dict(db_name, pred_sql)
#             my_sql = sql_parser.dict_to_raw_sql(db_name, pred_sql_dict)
#         except:#else:
#             error_convert += 1
#             my_sql = ''
#
#         # label = sql_parser.check_equal(sql_dict, my_sql_d)
#         try:
#             label = sql_parser.check_equal_script(db_name, my_sql, raw_sql)
#         except:
#             error_check += 1
#             label = False
#
#         if label is True:
#             correct += 1
#         else:
#             wrong += 1
#             # print(i)
#             # print(pred_sql)
#             # print(raw_sql)
#             # print()
#
#     print(f'#Case:{len(pred_result)}')
#     print(f'Error convert: {error_convert}')
#     print(f'Error check: {error_check}')
#     print(f'Correct: {correct}')
#     print(f'Wrong: {wrong}')