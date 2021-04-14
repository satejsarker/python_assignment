import logging
import pandas
from data_model import CommonCsvModel

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

test_data = pandas.read_csv('./test.csv')
train_data = pandas.read_csv('./train.csv')
ideal_data = pandas.read_csv('./ideal.csv')


class TrainDataset(object):
    pass


class TestDataset(object):
    pass


class IdealDataset(object):
    pass


if __name__ == '__main__':
    train_data_set = CommonCsvModel(data=train_data, mapper_cls=TrainDataset,
                                    table_name="train_data")
    print(train_data_set.get_row_wise_data())
    ideal_data_set = CommonCsvModel(data=ideal_data, mapper_cls=IdealDataset,
                                    table_name="ideal_data")
    # print(ideal_data_set.get_row_wise_data())
    test_data_set = CommonCsvModel(mapper_cls=TestDataset, table_name="test_data_set",
                                   data=test_data)
    print(test_data_set.get_row_wise_data())
