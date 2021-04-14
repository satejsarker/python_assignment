import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import create_session
from sqlalchemy import Table, Column, Integer, MetaData, FLOAT, exc
from sqlalchemy.orm import mapper

DB_URL = "sqlite:///./data1.db"
ENGINE = create_engine(DB_URL)
session = create_session(bind=ENGINE, autocommit=True, autoflush=True)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class CommonCsvModel:
    def __init__(self, data,table_name, mapper_cls, table_config=None):
        self.data = data
        self.engine = ENGINE
        self.mapper_cls = mapper_cls
        self.metadata = MetaData(bind=self.engine)
        self.table_name = table_name
        self.table = table_config or Table(self.table_name, self.metadata, Column('id', Integer, primary_key=True),
                                           *(Column(column, FLOAT) for column in self.csv_columns_list))
        mapper(self.mapper_cls, self.table)
        self.session = session
        if self.table_name not in self.engine.table_names():
            LOGGER.warning("New Table is creating")
            self.create_table()
            self.insert_data()

    @property
    def csv_columns_list(self) -> list:
        """
        csv columns list
        :return: list of column
        :rtype: list(str)
        """
        return list(self.data.columns.tolist())

    def create_table(self):
        """
        :return: None

        """
        return self.metadata.create_all()

    def insert_data(self):
        """
        insert data into database
        :return:
        """
        try:
            insert_data = []
            for i, data in enumerate(self.csv_data_to_model()):
                _obj = self.mapper_cls()
                setattr(_obj, 'id', i)
                for index, col in enumerate(self.csv_columns_list):
                    setattr(_obj, col, data[index])
                insert_data.append(_obj)
            self.session.bulk_save_objects(objects=insert_data)
            LOGGER.warning("inserted all data in {}".format(self.table_name))
        except exc.IntegrityError:
            LOGGER.error("data already added in {} table ".format(self.table_name))

    def csv_data_to_model(self):
        """
        CSV data extraction from row

        :return: list of  data from csv row
        :rtype: list
        """
        csv_data = []
        for index, row in self.data.iterrows():
            csv_data.append(row.values.tolist())
        return csv_data

    def table_all_data(self) -> list:
        """
        select all data from table
        :return: all the data from table
        :rtype list
        """
        return self.session.query(self.mapper_cls).all()

    def get_row_wise_data(self) -> list:
        """
        table row wise data with key
        :return: list of object
        :rtype list
        """
        all_data = []
        for data in self.table_all_data():
            row = {}
            for column in self.csv_columns_list:
                row.update({
                    column: getattr(data, column)
                })
            all_data.append(row)
        return all_data
