from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Article(Base):
    """SQLAlchemy model for the articles table."""
    __tablename__ = 'articles'

    id = Column(Integer, primary_key=True, autoincrement=False)  # Use `autoincrement=False` for external IDs
    title = Column(String, nullable=False)
    description = Column(String, nullable=True)
    published_time = Column(DateTime, nullable=True)
    word_count = Column(Integer, nullable=True)
    authors = Column(String, nullable=True)
    section_id = Column(String, nullable=True)