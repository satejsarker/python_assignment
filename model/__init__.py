from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, mapper, create_session

DB_URL = "sqlite:///../data.db"
ENGINE = create_engine(DB_URL)
# Session = sessionmaker(bind=ENGINE)
session = create_session(bind=ENGINE, autocommit=False, autoflush=True)
