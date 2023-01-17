# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Spider: A Large-Scale Human-Labeled Dataset for Text-to-SQL Tasks"""


import json
from third_party.spider.preprocess.get_tables import dump_db_json_schema

import datasets

from seq2seq.lf_util.sql_dict_parser import SqlParser
# from seq2seq.lf_util.sql_dict_parser_semql import SqlParser


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
dirty_sql_from_synthetic
}
"""

_DESCRIPTION = """\
dirty_sql_from_synthetic
"""

_HOMEPAGE = "https://yale-lily.github.io/spider"

_LICENSE = "CC BY-SA 4.0"

_URL = "https://drive.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0"


class DirtySQL(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="dirty_sql_from_synthetic",
            version=VERSION,
            description="Spider: A Large-Scale Human-Labeled Dataset for Text-to-SQL Tasks",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs):
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()
        #self.sql_parser = SqlParser('data/spider/tables.json', db_dir='data/spider/database')
        
        print("process dirty_sql_from_synthetic!!!!")
        
        

    def _info(self):
        features = datasets.Features(
            {
                "raw_query": datasets.Value("string"),
                "query": datasets.Value("string"),
                "question": datasets.Value("string"),
                "db_id": datasets.Value("string"),
                "db_path": datasets.Value("string"),
                "db_table_names": datasets.features.Sequence(datasets.Value("string")),
                "db_column_names": datasets.features.Sequence(
                    {
                        "table_id": datasets.Value("int32"),
                        "column_name": datasets.Value("string"),
                    }
                ),
                "db_column_types": datasets.features.Sequence(datasets.Value("string")),
                "db_primary_keys": datasets.features.Sequence({"column_id": datasets.Value("int32")}),
                "db_foreign_keys": datasets.features.Sequence(
                    {
                        "column_id": datasets.Value("int32"),
                        "other_column_id": datasets.Value("int32"),
                    }
                ),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        #downloaded_filepath =  dl_manager.download_and_extract(_URL)
        downloaded_filepath = 'cache/dirty_sql/'
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepath": downloaded_filepath + "aug_examples_train.json",
                    "db_path": "",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepath": downloaded_filepath + "aug_examples_dev.json",
                    "db_path": "",
                },
            ),
        ]

    def _generate_examples(self, data_filepath, db_path):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", data_filepath)
        print("read examples from path: ", data_filepath)
        with open(data_filepath, encoding="utf-8") as f:
            spider = [json.loads(line) for line in f.readlines()]

            for idx, sample in enumerate(spider):
                db_id = sample["db_id"]

                yield idx, {
                    "raw_query": sample["raw_query"],
                    "query": sample["query"],
                    "question": sample["raw_query"],
                    "db_id": db_id,
                    "db_path": db_path,
                    "db_table_names": sample["db_table_names"],
                    "db_column_names": {"table_id": sample["db_column_names"][0], "column_name": sample["db_column_names"][1]},
                    "db_column_types": sample["col_type"],
                    "db_primary_keys": [],
                    "db_foreign_keys": []
                }
