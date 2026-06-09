from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from os import path
from configs import DATA_DIR
from tables import Base


class Database:
    def __init__(self, db_dir=DATA_DIR, name='stock_db'):
        name = name if name[-3:] == '.db' else name + '.db'
        db_url = f'sqlite:////{path.join(db_dir, name)}'
        self.engine = create_engine(db_url)
        self.session_factory = sessionmaker(bind=self.engine)

    def create_tables(self):
        Base.metadata.create_all(self.engine)
        print("Tables created (if they didn't exist).")

    def get_session(self) -> Session:
        return self.session_factory()

    def dispose(self):
        self.engine.dispose()

