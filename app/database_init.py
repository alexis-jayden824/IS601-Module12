from app.database import engine
from app.models.user import Base

def init_db(bind_engine=None):
    Base.metadata.create_all(bind=bind_engine or engine)

def drop_db(bind_engine=None):
    Base.metadata.drop_all(bind=bind_engine or engine)

if __name__ == "__main__":
    init_db() # pragma: no cover