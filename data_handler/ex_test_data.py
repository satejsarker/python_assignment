"""
Test data class for query and input
"""
from model.common import CommonCsvModel
from data_handler import test_data
from model import session, ENGINE


class ExTestData(CommonCsvModel):
    """
    Train data class inherited from common CSV model class
    """

    def __init__(self):
        self.data = test_data
        self.table_name = "test_data"
        self.engine = ENGINE
        self.session = session
        super().__init__(self.data,
                         self.engine, session,
                         table_name=self.table_name)
        super().create_table()
        super().insert_data()


if __name__ == '__main__':
    t = ExTestData()
