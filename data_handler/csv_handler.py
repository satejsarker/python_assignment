import pandas

from model import session, ENGINE
# from model.schema import Test
from model.common import CommonCsvModel, Dataset

# data = pandas.read_csv('../data/test.csv')
# objs = []
# for index, row in data.iterrows():
#     objs.append(Test(row['x'], row['y']))
#
# session.bulk_save_objects(objects=objs)
# session.commit()

test_data = pandas.read_csv('../data/test.csv')
train_data = pandas.read_csv('../data/train.csv')
ideal_data = pandas.read_csv('../data/ideal.csv')

# train data
# TODO : create class for train data and move all the function to that class
train_data_col = list(train_data.columns.values.tolist())
# create table
Train_table = CommonCsvModel(data=train_data, engine=ENGINE, session=session, table_name='train_table')

Train_table.create_table()
# Train_table.insert_data()
