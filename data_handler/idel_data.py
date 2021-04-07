"""
Ideal data class for query and input
"""
from model.common import CommonCsvModel
from data_handler import ideal_data
from model import session, ENGINE


class IdealData(CommonCsvModel):
    """
    Train data class inherited from common CSV model class
    """

    def __init__(self):
        self.data = ideal_data
        self.table_name = "ideal_data"
        self.engine = ENGINE
        self.session = session
        super().__init__(self.data,
                         self.engine, session,
                         table_name=self.table_name)
        super().create_table()
        super().insert_data()


if __name__ == '__main__':
    t = IdealData()
    print(t.get_row_wise_data())
