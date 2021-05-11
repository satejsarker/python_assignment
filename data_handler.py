import logging
from copy import deepcopy
from itertools import chain
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas
from sqlalchemy import Table, Column, MetaData, FLOAT, exc, PrimaryKeyConstraint
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import create_session
from sqlalchemy.orm import mapper

train_data = pandas.read_csv('./train.csv')
ideal_data = pandas.read_csv('./ideal.csv')
test_data = pandas.read_csv('./test.csv')
DB_URL = "sqlite:///./data1.db"
ENGINE = create_engine(DB_URL)
session = create_session(bind=ENGINE, autocommit=True, autoflush=True)
metadata = MetaData(bind=ENGINE)
Base = declarative_base(metadata=metadata)

LOGGER = logging.getLogger("Python assigment")
logging.basicConfig(level=logging.INFO)


class TrainDataset(object):
    pass


class IdealDataset(object):
    pass


class DataVisualization:

    @staticmethod
    def ols_data(column_a, column_b, file_name):
        _A = np.vstack([column_a, np.ones(len(column_a))]).T
        m, c = np.linalg.lstsq(_A, column_b, rcond=None)[0]
        _ = plt.plot(column_a, column_b, '*', label='train data', markersize=10)
        _ = plt.plot(column_a, m * column_a + c, 'r', label='ideal data')
        plt.legend()
        plt.savefig(f"{file_name}.png")
        plt.show()


class TestMapperDataset(Base):
    __tablename__ = "mapped_test_data"
    x = Column(FLOAT, primary_key=True)
    y = Column(FLOAT, primary_key=True)
    ideal_fn_data = Column(FLOAT)
    deviation = Column(FLOAT)
    __table_args__ = (
        PrimaryKeyConstraint("x", "y"),
    )

    @staticmethod
    def insert_data(all_data: List[dict]) -> None:
        try:
            session.bulk_save_objects(objects=all_data)
        except exc.IntegrityError:
            LOGGER.warning("data already added in table ")


class CommonModel:
    def __init__(self, table_name: str, table_columns: list, mapper_cls, table_config=None):
        self.table_name = table_name
        self.table_columns = table_columns
        self.mapper_class = mapper_cls
        self.engine = ENGINE
        self.metadata = metadata
        self.session = session
        if "x" in table_columns:
            table_columns = deepcopy(self.table_columns)
            table_columns.remove("x")
        if table_config is not None:
            self.table = table_config
        else:
            self.table = table_config or Table(self.table_name, self.metadata, Column('x', FLOAT, primary_key=True),
                                               *(Column(column, FLOAT) for column in table_columns))
        mapper(self.mapper_class, self.table)

    def create_table(self) -> None:
        """
        :return: None

        """
        try:
            return self.metadata.create_all()
        except:
            LOGGER.warning("Table not created")

    def insert_data(self, all_data: list) -> None:
        """

        :param all_data: all object data list
        :type all_data: list
        :return: None
        """
        try:
            LOGGER.warning("inserted all data in {}".format(self.table_name))
            return self.session.bulk_save_objects(objects=all_data)
        except exc.IntegrityError as err:
            LOGGER.warning("data already added in {} table ".format(self.table_name))

    def data_mapper(self, all_data: list) -> List[dict]:
        """
        Map data list to mapper class for insert

        :param all_data: list of all data
        :type all_data:list
        :return: data object list
        :rtype: List[dict]
        """
        mapped_data = []
        for i, data in enumerate(all_data):
            _obj = self.mapper_class()
            for index, column in enumerate(self.table_columns):
                setattr(_obj, column, data[index])
                mapped_data.append(_obj)
        return mapped_data

    @property
    def table_exists(self):
        """

        :return: True if table exists
        """
        table_exists = self.table_name in self.engine.table_names()
        if table_exists:
            LOGGER.warning("table exits")
            return True
        return False

    def get_column_data(self, column: str) -> list:
        """

        :param column: table column name
        :type column: str
        :return: column numpy array
        :rtype: list
        """
        all_value = self.session.query(getattr(self.mapper_class, column)).all()
        return np.array(list(chain(*all_value)))

    def get_fn_val_by_input(self, x_input: float, y_idea: str) -> float:
        """
        get corresponding y value from table for input x
        :param x_input: x value for search
        :param y_idea: y column name
        :return: function value for corresponding x
        :rtype: float
        """
        return (self.session.query(getattr(self.mapper_class, y_idea))
            .where(getattr(self.mapper_class, "x") == x_input)
            .one()[0])

    def table_all_data(self) -> list:
        """
        select all data from table
        :return: all the data from table
        :rtype list
        """
        return self.session.query(self.mapper_class).all()


class CommonCsvModel(CommonModel):
    def __init__(self, data, table_name=None, mapper_cls=None):
        self.data = data
        table_column = self.data.columns.tolist()
        table_column.remove("x")
        if table_name and mapper_cls:
            super().__init__(
                table_name=table_name,
                table_columns=table_column,
                mapper_cls=mapper_cls

            )

    @property
    def csv_columns_list(self) -> list:
        """
        csv columns list
        :return: list of column
        :rtype: list(str)
        """
        return list(self.data.columns.tolist())

    def insert_from_csv_data(self):
        """
        insert data into database
        :return:
        """
        insert_data = []
        for i, data in enumerate(self.csv_data()):
            _obj = self.mapper_class()
            for index, col in enumerate(self.csv_columns_list):
                setattr(_obj, col, data[index])
            insert_data.append(_obj)
        self.insert_data(all_data=insert_data)

    def csv_data(self):
        """
        CSV data extraction from row

        :return: list of  data from csv row
        :rtype: list
        """
        csv_data = []
        for index, row in self.data.iterrows():
            csv_data.append(row.values.tolist())
        return csv_data


if __name__ == '__main__':
    train_fn = CommonCsvModel(
        table_name="train",
        data=train_data,

        mapper_cls=TrainDataset)
    if not train_fn.table_exists:
        train_fn.create_table()
        train_fn.insert_from_csv_data()
    else:
        train_fn.insert_from_csv_data()
    ideal_fn = CommonCsvModel(
        table_name="ideal",
        data=ideal_data,
        mapper_cls=IdealDataset)
    if not ideal_fn.table_exists:
        ideal_fn.create_table()
        ideal_fn.insert_from_csv_data()
    else:
        ideal_fn.insert_from_csv_data()
    ideal_function_list = ideal_fn.csv_columns_list
    ideal_function_list.remove("x")
    train_fn_function_list = train_fn.csv_columns_list
    train_fn_function_list.remove("x")
    chosen = []
    metadata.create_all(bind=ENGINE)
    # print(ideal_fn.get_fn_val_by_input(x_input=19.5, y_idea="y39"))
    for t_column in train_fn_function_list:
        chosen_fn = None
        train_data = train_fn.get_column_data(t_column)  # y1,y2,y3,y4
        min_sum_sqr = None
        max_div = None
        for i_column in ideal_function_list:
            ideal_data = ideal_fn.get_column_data(i_column)

            deviation = np.absolute(train_data - ideal_data)  # delta
            system_error = np.square(deviation)
            max_deviation = deviation.max()  # max  delta value (yi-yi^)^2
            sum_delta_error = system_error.sum()  # sum of delta
            if max_div is not None and min_sum_sqr is not None:
                if min_sum_sqr >= sum_delta_error:
                    min_sum_sqr = sum_delta_error
                    chosen_fn = i_column
                if max_deviation >= max_div:
                    max_div = max_deviation
            else:
                min_sum_sqr = sum_delta_error
                max_div = max_deviation
        chosen.append((t_column, chosen_fn, max_div))
    LOGGER.info("chosen function {}".format(chosen))
    # visualizations after OLS of train data with ideal data
    for chosen_data in chosen:
        train_data, chosen_ideal_fn, max_deviation = chosen_data
        ideal_data = ideal_fn.get_column_data(chosen_ideal_fn)
        train_data = train_fn.get_column_data(train_data)
        DataVisualization.ols_data(train_data, ideal_data, chosen_ideal_fn)
    test_data = CommonCsvModel(
        data=test_data,
        mapper_cls=None
    )
    # creating mapped test table
    t = TestMapperDataset()
    test_data_deviation = []
    for test in test_data.csv_data():
        x, y = test
        for chosen_data in chosen:
            train_data, chosen_ideal_fn, max_deviation = chosen_data
            ideal_mapped_data = ideal_fn.get_fn_val_by_input(x, chosen_ideal_fn)
            deviation = np.absolute(y - ideal_mapped_data)
            if deviation <= (max_deviation * np.sqrt(2)):
                obj = TestMapperDataset(x=x, y=y, ideal_fn_data=ideal_mapped_data, deviation=deviation)
                # test_data_deviation.append([x, y, ideal_mapped_data, deviation])
                test_data_deviation.append(obj)
                break
            else:
                LOGGER.info(f"test data {x} cant be mapped")
    LOGGER.info(f"number of mapped test data {len(test_data_deviation)}")

    TestMapperDataset.insert_data(test_data_deviation)
