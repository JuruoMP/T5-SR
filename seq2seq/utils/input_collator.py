from collections import defaultdict
import re
import torch


CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group by', 'order by', 'limit', 'intersect', 'union', 'except', 'having')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')

COND_OPS = ('and', 'or')
ORDER_OPS = ('desc', 'asc')

OTHER_KEYWORDS = ['(', ')', '*', ',', 'order', 'group', 'by', 'distinct']

def is_number(s):
    try:
        float(s)
        return True
    except:
        return False


def sql_split(text):
    tokens = text.replace(',', ' ,').replace('  ', ' ').split()
    new_tokens = []
    agg_token = []
    for token in tokens:
        if agg_token:
            agg_token.append(token)
            if agg_token[-1][-1] == agg_token[0][0]:
                new_tok = ' '.join(agg_token)
                new_tokens.append(new_tok)
                agg_token = []
        else:
            if token[0] not in ('"', "'"):
                new_tokens.append(token)
            elif token[-1] == token[0]:
                new_tokens.append(token)
            else:
                agg_token.append(token)
    assert agg_token == []
    return new_tokens


class ConstrainedInputCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.sql_keywords = list(CLAUSE_KEYWORDS) + list(JOIN_KEYWORDS) + list(WHERE_OPS) + list(UNIT_OPS) \
                            + list(AGG_OPS) + list(COND_OPS) + list(ORDER_OPS) + list(OTHER_KEYWORDS)
        self.sql_keywords = [x.upper() for x in self.sql_keywords]
        self.dot_id = self.tokenize_with_id('.')[1]

    @staticmethod
    def normalize(query: str) -> str:
        def comma_fix(s):
            # Remove spaces in front of commas
            return s.replace(" , ", ", ")

        def white_space_fix(s):
            # Remove double and triple spaces
            return " ".join(s.split())

        def lower(s):
            # Convert everything except text between (single or double) quotation marks to lower case
            return re.sub(r"\b(?<!['\"])(\w+)(?!['\"])\b", lambda match: match.group(1).lower(), s)

        return comma_fix(white_space_fix(lower(query)))

    def encode_utt(self, db_id, input_text, serialized_schema, output_text, max_source_length, max_target_length):
        table_names, column_names = [], []
        cur_table, table_column_names = None, []
        schema_raw_tokens = serialized_schema.split()
        print("schema_raw_tokens:")
        print(schema_raw_tokens)
        for i, token in enumerate(schema_raw_tokens):
            if token == ':':
                cur_table = schema_raw_tokens[i - 1]
                table_names.append(cur_table)
            elif token == ',':
                cur_col = schema_raw_tokens[i - 1]
                column_names.append((cur_table, cur_col))

        input_text_indices = self.tokenizer.encode(input_text)
        eos_id = input_text_indices[-1]
        input_text_indices = input_text_indices[:-1]
        serialized_schema_indices = self.tokenizer.encode(serialized_schema)[:-1]
        tab_separator_id = self.tokenizer.convert_tokens_to_ids(':')
        col_separator_id = self.tokenizer.convert_tokens_to_ids(',')
        tab_indicator_indices = \
            [-1] + [i + 1 + len(input_text_indices)
                    for i, tok_id in enumerate(serialized_schema_indices) if tok_id == tab_separator_id]
        col_indicator_indices = \
            [-1] + [i + 1 + len(input_text_indices)
                    for i, tok_id in enumerate(serialized_schema_indices) if tok_id == col_separator_id]
        input_indices = input_text_indices + serialized_schema_indices + [eos_id]
        assert self.tokenizer.encode(f'{input_text} {serialized_schema}') == input_indices
        if len(input_indices) > max_source_length:
            input_indices = input_indices[:max_source_length]
            input_indices[-1] = eos_id
        assert self.tokenizer.encode(f'{input_text} {serialized_schema}', max_length=max_source_length) == input_indices
        input_dict = {
            'input_ids': input_indices,
            'attention_mask': [1 for _ in range(len(input_indices))],
            'table_indices': tab_indicator_indices,
            'column_indices': col_indicator_indices,
        }

        output_token_list = sql_split(self.normalize(output_text))
        token_id_list, token_type_list = [], []
        db_name_ids = self.tokenize_with_id(db_id + ' | ')
        token_id_list += db_name_ids
        token_type_list += ['db' for _ in range(len(token_id_list))]
        # todo: value should not be lowered
        for token in output_token_list:
            if '"' not in token and '.' not in token:
                if token.upper() in self.sql_keywords:
                    token = token.lower()
                    token_ids = self.tokenize_with_id(token)
                    token_id_list += token_ids
                    token_type_list += ['keyword'] * len(token_ids)
                elif is_number(token):
                    token_ids = self.tokenize_with_id(token)
                    token_id_list += token_ids
                    token_type_list += ['value'] * len(token_ids)
                else:
                    table_id = table_names.index(token)
                    token = token.lower()
                    token_ids = self.tokenize_with_id(token)
                    token_id_list += token_ids
                    token_type_list += [f'B-table-{table_id}'] + [f'I-table-{table_id}'] * (len(token_ids) - 1)
            elif is_number(token):  # e.g. 20.0
                token_ids = self.tokenize_with_id(token)
                token_id_list += token_ids
                token_type_list += ['value'] * len(token_ids)
            elif '"' not in token:
                assert '.' in token
                token = token.lower()
                table_name, column_name = token.strip(',').split('.')
                
                print(table_names)
                print(column_names)
                table_id, column_id = table_names.index(table_name), column_names.index((table_name, column_name))
                token_ids = self.tokenize_with_id(token)
                dot_id_index = token_ids.index(self.dot_id)
                token_id_list += token_ids
                token_type_list += [f'B-table-{table_id}'] + [f'I-table-{table_id}'] * (dot_id_index - 1) + ['dot'] + \
                                   [f'B-column-{column_id}'] + [f'I-column-{column_id}'] * (len(token_ids) - dot_id_index - 2)
            else:
                # token = token.lower()  # notice: values are lowered
                assert token.strip()[0] in ('"', "'") and token.strip()[-1] in ('"', "'")
                token_ids = self.tokenize_with_id(token)
                token_id_list += token_ids
                token_type_list += ['value'] * len(token_ids)
        token_id_list.append(self.tokenizer.eos_token_id)
        token_type_list.append('none')
        # assert self.tokenizer.encode(db_id + ' | ' + output_text.lower()) == token_id_list
        if len(token_id_list) > max_target_length:
            n_token_eliminate = max_target_length - len(token_id_list)
            token_id_list = token_id_list[n_token_eliminate:]
            token_type_list = token_type_list[n_token_eliminate:]
        # assert self.tokenizer.encode(db_id + ' | ' + output_text.lower(), max_length=max_target_length) == token_id_list

        output_dict = {
            'labels': token_id_list,
            # 'attention_mask': [1 for _ in range(len(token_id_list))],
            # 'constrained_lm_labels': None,
            'table_labels': [int(tok_type[8:]) + 1 if tok_type.startswith('B-table-') else 0 for tok_type in token_type_list],
            'column_labels': [int(tok_type[9:]) + 1 if tok_type.startswith('B-column-') else 0 for tok_type in token_type_list]
        }
        assert any(x != -100 for x in output_dict['table_labels'])
        return input_dict, output_dict

    def debug_encode(self, batch, max_source_length, max_target_length):
        from seq2seq.utils.dataset import DataTrainingArguments, normalize, serialize_schema
        def spider_get_input(
                question: str,
                serialized_schema: str,
                prefix: str,
        ) -> str:
            return prefix + question.strip() + " " + serialized_schema.strip()

        prefix = ''
        inputs = [
            spider_get_input(question=question, serialized_schema=serialized_schema, prefix=prefix)
            for question, serialized_schema in zip(batch["question"], batch["serialized_schema"])
        ]

        def spider_get_target(
                query: str,
                db_id: str,
                normalize_query: bool,
                target_with_db_id: bool,
        ) -> str:
            _normalize = normalize if normalize_query else (lambda x: x)
            return f"{db_id} | {_normalize(query)}" if target_with_db_id else _normalize(query)

        model_inputs: dict = self.tokenizer(
            inputs,
            max_length=max_source_length,
            padding=False,
            truncation=True,
            return_overflowing_tokens=False,
        )

        targets = [
            spider_get_target(
                query=query,
                db_id=db_id,
                normalize_query=True,#data_training_args.normalize_query,
                target_with_db_id=True,#data_training_args.target_with_db_id,
            )
            for db_id, query in zip(batch["db_id"], batch["query"])
        ]

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=max_target_length,
                padding=False,
                truncation=True,
                return_overflowing_tokens=False,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def batch_encode(self, batch, max_source_length, max_target_length):
        db_id_list = batch['db_id']
        input_text_list = batch['question']
        serialized_schema_list = batch['serialized_schema']
        output_text_list = batch['query']
        input_dict_list, output_dict_list = [], []
        for db_id, input_text, serialized_schema, output_text in \
                zip(db_id_list, input_text_list, serialized_schema_list, output_text_list):
            input_dict, output_dict = self.encode_utt(db_id, input_text, serialized_schema, output_text,
                                                      max_source_length, max_target_length)
            input_dict_list.append(input_dict)
            output_dict_list.append(output_dict)
        input_dict = {k: [x[k] for x in input_dict_list] for k in input_dict_list[0].keys()}
        output_dict = {k: [x[k] for x in output_dict_list] for k in output_dict_list[0].keys()}
        ret_dict = input_dict.copy()
        ret_dict.update(output_dict)
        # debugging
        # debug_dict = self.debug_encode(batch, max_source_length, max_target_length)
        # for key in ('input_ids', 'attention_mask', 'labels'):
        #     assert debug_dict[key] == ret_dict[key]

        return ret_dict

    def tokenize_with_id(self, token):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(token))

    def collate_fn(self, batch_dict):
        ret_dict = {}

        max_input_ids_len = max(len(x['input_ids']) for x in batch_dict)
        max_table_indices_len = max(len(x['table_indices']) for x in batch_dict)
        max_column_indices_len = max(len(x['column_indices']) for x in batch_dict)
        max_output_ids_len = max(len(x['labels']) for x in batch_dict)
        max_output_table_len = max(len(x['table_labels']) for x in batch_dict)
        max_output_column_len = max(len(x['column_labels']) for x in batch_dict)

        for x in batch_dict:
            x['table_indices'] = [_ if _ < 512 else 0 for _ in x['table_indices']]
            x['column_indices'] = [_ if _ < 512 else 0 for _ in x['column_indices']]

        pad_id = self.tokenizer.pad_token_id
        ret_dict['input_ids'] = [x['input_ids'] + [pad_id] * (max_input_ids_len - len(x['input_ids'])) for x in batch_dict]
        ret_dict['attention_mask'] = [x['attention_mask'] + [0] * (max_input_ids_len - len(x['attention_mask'])) for x in batch_dict]
        ret_dict['table_indices'] = [x['table_indices'] + [pad_id] * (max_table_indices_len - len(x['table_indices'])) for x in batch_dict]
        ret_dict['table_indices_mask'] = [[1] * len(x['table_indices']) + [pad_id] * (max_table_indices_len - len(x['table_indices'])) for x in batch_dict]
        ret_dict['column_indices'] = [x['column_indices'] + [pad_id] * (max_column_indices_len - len(x['column_indices'])) for x in batch_dict]
        ret_dict['column_indices_mask'] = [[1] * len(x['column_indices']) + [pad_id] * (max_column_indices_len - len(x['column_indices'])) for x in batch_dict]
        # ret_dict['decoder_input_ids'] = [[decoder_start_id] + x['decoder_input_ids'][:-1] + [pad_id] * (max_output_ids_len - len(x['decoder_input_ids'])) for x in batch_dict]
        # ret_dict['decoder_attention_mask'] = [[1] * len(x['decoder_ipnut_ids']) + [0] * (max_output_ids_len - len(x['decoder_input_ids'])) for x in batch_dict]
        ret_dict['labels'] = [x['labels'] + [-100] * (max_output_ids_len - len(x['labels'])) for x in batch_dict]
        ret_dict['constrained_lm_labels'] = ret_dict['labels']  # todo
        ret_dict['table_labels'] = [x['table_labels'] + [-100] * (max_output_table_len - len(x['table_labels'])) for x in batch_dict]
        ret_dict['column_labels'] = [x['column_labels'] + [-100] * (max_output_column_len - len(x['column_labels'])) for x in batch_dict]

        return {k: torch.tensor(ret_dict[k], dtype=torch.long) for k in ret_dict}
