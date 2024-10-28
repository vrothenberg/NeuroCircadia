# chainlit-app/models.py 

from sqlalchemy import Column, String, Boolean, Text, ForeignKey, Integer, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True)
    identifier = Column(String, unique=True, nullable=False)
    user_metadata = Column(Text, nullable=False)  # Keep other metadata here if needed
    password = Column(Text, nullable=False)  # New field for password
    createdAt = Column(Text)


class Thread(Base):
    __tablename__ = "threads"
    id = Column(String, primary_key=True)
    createdAt = Column(Text)
    name = Column(Text)
    userId = Column(String, ForeignKey("users.id"))
    userIdentifier = Column(String)
    tags = Column(Text)
    thread_metadata = Column(Text)  # Renamed from 'metadata'
    steps = relationship("Step", backref="thread", cascade="all, delete")


class Step(Base):
    __tablename__ = "steps"
    id = Column(String, primary_key=True)
    name = Column(Text, nullable=False)
    type = Column(Text, nullable=False)
    threadId = Column(String, ForeignKey("threads.id"), nullable=False)
    parentId = Column(String)
    disableFeedback = Column(Boolean, nullable=False)
    streaming = Column(Boolean, nullable=False)
    waitForAnswer = Column(Boolean)
    isError = Column(Boolean)
    step_metadata = Column(Text)  # Renamed from 'metadata'
    tags = Column(Text)
    input = Column(Text)
    output = Column(Text)
    createdAt = Column(Text)
    start = Column(Text)
    end = Column(Text)
    generation = Column(Text)
    showInput = Column(Text)
    language = Column(Text)
    indent = Column(Integer)


class Feedback(Base):
    __tablename__ = "feedbacks"
    id = Column(String, primary_key=True)
    forId = Column(String, nullable=False)
    threadId = Column(String, ForeignKey("threads.id"), nullable=False)
    value = Column(Integer, nullable=False)
    comment = Column(Text)
