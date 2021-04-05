"""
Train data class for query and input
"""
from model.common import CommonCsvModel
from data_handler import train_data
from model import session, ENGINE


class TrainData(CommonCsvModel):
    """
    Train data class inherited from common CSV model class
    """

    def __init__(self):
        self.data = train_data
        self.table_name = "train_data"
        self.engine = ENGINE
        self.session = session
        super().__init__(self.data,
                         self.engine, session,
                         table_name=self.table_name)
        super().create_table()
        super().insert_data()


if __name__ == '__main__':
    t = TrainData()
    data = t.get_row_wise_data()
    print(data)
