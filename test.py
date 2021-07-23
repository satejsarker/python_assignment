"""
Unite test for data_handler.py
"""
import os
from typing import ByteString

import pandas
from _pytest.compat import cached_property

from data_handler import CommonCsvModel


class TestCommonCsvModel:
    @cached_property
    def dummy_csv(self) -> ByteString:
        """
        Read dummy csv file
        :return: csv file
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        _csv = os.path.join(current_dir, 'unitest_file.csv')
        return pandas.read_csv(_csv)

    @cached_property
    def csv_model(self) -> CommonCsvModel:
        """
        Create instance of CommonCsvModel

        :return: instance of CommonCsvModel
        :rtype CommonCsvModel
        """
        csv_model = CommonCsvModel(data=self.dummy_csv)
        return csv_model

    def test_csv_columns_list(self):
        """
        Test csv columns list

        """
        assert type(self.csv_model.csv_columns_list()) == list
        assert self.csv_model.csv_columns_list() == ["x", "y"]