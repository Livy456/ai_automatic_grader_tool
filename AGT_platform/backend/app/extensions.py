from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

#Base = declarative_base()
engine = None
SessionLocal = sessionmaker(autocommit=False, autoflush=False)


def init_db(database_url: str):

    global engine, SessionLocal
    print("database url (inside extensions.py): ", database_url)
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)

    print("updated database url (inside extensions.py): ", database_url)
    
    engine = create_engine(database_url, pool_pre_ping=True)
    SessionLocal.configure(bind=engine)

    return engine