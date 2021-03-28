from sqlalchemy import MetaData, Table, Column, Integer
from sqlalchemy.orm import  mapper
from model import ENGINE
metadata = MetaData()

test = Table('test', metadata,
             Column("id", Integer, primary_key=True),
             Column("x", Integer),
             Column("y", Integer)
             )

# Mapper class
class Test():
    def __init__(self, x, y):
        self.x = x
        self.y = y

mapper(Test,test)
# metadata.create_all(ENGINE)
