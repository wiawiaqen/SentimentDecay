from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
from data.models import Article  
from data.get_news.database_connection import get_sql_server_connection

# Interpret the config file for Python logging.
if context.config.config_file_name is not None:
    fileConfig(context.config.config_file_name)

# Use the metadata from the Article model
target_metadata = Article.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = context.config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = get_sql_server_connection()  # Now connects to SQLite

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
