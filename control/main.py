from model import Session
from model.schema import Test


class Logic:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.session = Session()

    def add(self):
        t1 = Test(self.x, self.y)
        self.session.add(t1)
        self.session.commit()
        print("db value added")


if __name__ == '__main__':
    l = Logic(1, 3)
    l.add()
