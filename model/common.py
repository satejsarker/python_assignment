import logging
from sqlalchemy import Table, Column, Integer, MetaData, FLOAT, exc
from sqlalchemy.orm import mapper

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class Dataset(object):
    pass


class CommonCsvModel:
    def __init__(self, colums, data, engine, session):
        self.colums = colums
        self.data = data
        self.engine = engine
        self.metadata = MetaData(bind=self.engine)
        self.table_name = 'train_data'
        self.table = Table(self.table_name, self.metadata, Column('id', Integer, primary_key=True),
                           *(Column(column, FLOAT) for column in self.colums))
        mapper(Dataset, self.table)
        self.session = session

    def create_table(self):
        """
        create table if doesn't exists
        :return: None

        """
        if self.table_name not in self.engine.table_names():
            LOGGER.warning("New Table is creating")
            return self.metadata.create_all()
        else:
            LOGGER.warning("table is already there ")
            return None

    def insert_data(self):
        """
        insert data into database
        :return:
        """
        try:
            insert_data = []
            for i, data in enumerate(self.csv_data_to_model()):
                _obj = Dataset()
                setattr(_obj, 'id', i)
                for index, col in enumerate(self.colums):
                    setattr(_obj, col, data[index])
                insert_data.append(_obj)
            self.session.bulk_save_objects(objects=insert_data)
            LOGGER.warning("inserted all data in {}".format(self.table_name))
        except exc.IntegrityError:
            LOGGER.error("data already added in {} table ".format(self.table_name))

    def csv_data_to_model(self):
        csv_data = []
        for index, row in self.data.iterrows():
            csv_data.append(row.values.tolist())
        return csv_data
