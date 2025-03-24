from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

def get_sql_server_connection() -> Engine:
    """Create a connection to the SQLite database (temporary).

    Returns:
        Engine: SQLAlchemy engine connected to the SQLite database.
    """
    connection_string = "sqlite:///d:/thesis/implementation/temp_database.db"
    engine = create_engine(connection_string)
    return engine
