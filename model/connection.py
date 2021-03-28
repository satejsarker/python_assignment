from sqlalchemy import create_engine,column
from sqlalchemy.orm import scoped_session, sessionmaker, relation
engine=create_engine("sqlite:///../data.db")
db_session=scoped_session(sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
))
Model = declarative_base(name='Model')
Model.query = db_session.query_property()

def init_db():
    Model.metadata.create_all(bind=engine)
    
