import logging
import pandas
import numpy
from data_model import CommonCsvModel

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

test_data = pandas.read_csv('./test.csv')
ideal_data = pandas.read_csv('./ideal.csv')


class Helper:
    """
    Helper class for common functions
    """

    @staticmethod
    def compare_with_fn(first_fn: float, second_fn: float) -> int:
        return pow((second_fn - first_fn), 2)

    @staticmethod
    def get_single_fn(column: list, data: object) -> object:
        column.remove("x")
        for _clm in column:
            _obj = {
                "x": data.get('x'),
                "fn": data.get(_clm)
            }
            yield _obj


class ChosenIdealFunction(object):
    pass


class TrainDataset(object):
    pass


class TestDataset(object):
    pass


class IdealDataset(object):
    pass


class ChosenData(CommonCsvModel, Helper):
    def __init__(self):
        self.train_data = pandas.read_csv('./train.csv')
        self.columns = list(self.train_data.columns.tolist())

        super().__init__(data=[], columns=self.columns,
                         table_name='chosen_ideal_function',
                         mapper_cls=ChosenIdealFunction)
        if self.table_exists:
            self.create_table()

    def chosen_data(self, data: list, input: float) -> None:
        columns = list(self.train_data.columns.tolist())
        data.insert(0, input)
        _obj = self.mapper_cls()
        for index, col in enumerate(columns):
            setattr(_obj, col, data[index])
        return _obj


class TrainData(CommonCsvModel, Helper):
    def __init__(self):
        self.train_data = pandas.read_csv('./train.csv')

        super().__init__(self.train_data, 'train', TrainDataset)
        if self.table_exists:
            self.insert_from_csv_data()
        else:
            self.create_table()


class IdealData(CommonCsvModel, Helper):
    def __init__(self):
        self.ideal_data = pandas.read_csv('./ideal.csv')

        super().__init__(self.ideal_data, 'ideal', IdealDataset)
        if self.table_exists:
            self.insert_from_csv_data()
        else:
            self.create_table()


if __name__ == '__main__':
    train_fn = TrainData()

    ideal_fn = IdealData()
    chosen_fn = ChosenData()
    ch_fn = []
    for tr_data in train_fn.get_row_wise_data():
        _all_compare = []
        for t_data in train_fn.get_single_fn(train_fn.csv_columns_list, tr_data):
            choosed_fn = []
            for ideal_data_set in ideal_fn.get_fn_wise_numpy_array():
                compare_arr = ideal_data_set - t_data["fn"]
                choosed_fn.append(numpy.power(compare_arr, 2))
            _all_compare.append(numpy.min(choosed_fn))
        ch_fn.append(chosen_fn.chosen_data(
            data=_all_compare,
            input=tr_data.get("x")))
    chosen_fn.insert_data(all_data=ch_fn)
    LOGGER.info("4 ideal function added for input")
