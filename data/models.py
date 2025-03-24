from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Article(Base):
    """SQLAlchemy model for the articles table."""
    __tablename__ = 'articles'

    id = Column(String, primary_key=True) 
    title = Column(String, nullable=False)
    description = Column(String, nullable=True)
    published_time = Column(DateTime, nullable=True)
    word_count = Column(Integer, nullable=True)
    authors = Column(String, nullable=True)
    section_id = Column(String, nullable=True)
    sentiment_label = Column(String, nullable=True)  # New column for sentiment label
    sentiment_score = Column(Float, nullable=True)   # New column for sentiment confidence score
    topic = Column(String, nullable=True)  # New column for topic
    topic_probability = Column(Float, nullable=True)  # New column for topic probability
