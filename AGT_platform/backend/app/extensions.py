#from sqlalchemy import create_engine
#from sqlalchemy.orm import sessionmaker, declarative_base

#engine = None
#SessionLocal = None
#Base = declarative_base()

#def init_db(database_url: str):
#    global engine, SessionLocal
#    engine = create_engine(database_url, pool_pre_ping=True)
#    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
# app/extensions.py






# from __future__ import annotations
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker, declarative_base
# from sqlalchemy.exc import NoSuchModuleError

# engine = None
# SessionLocal = None
# Base = declarative_base()

# def init_db(database_url: str):
#     global engine, SessionLocal

#     # Normalize common mistake
#     if database_url.startswith("postgres://"):
#         database_url = database_url.replace("postgres://", "postgresql://", 1)

#     try:
#         engine = create_engine(database_url, pool_pre_ping=True)
#     except NoSuchModuleError as e:
#         raise RuntimeError(
#             f"Database URL dialect not recognized or driver missing.\n"
#             f"Got DATABASE_URL={database_url}\n"
#             f"Use postgresql://... or postgresql+psycopg://...\n"
#             f"Install a driver: pip install 'psycopg[binary]'  (recommended) "
#             f"or pip install psycopg2-binary"
#         ) from e

#     SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
           

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False)
engine = None

def init_db(database_url: str):

    global engine, SessionLocal
    print("database url (inside extensions.py): ", database_url)
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)

    print("updated database url (inside extensions.py): ", database_url)
    
    engine = create_engine(database_url, pool_pre_ping=True)
    SessionLocal.configure(bind=engine)

    return engine