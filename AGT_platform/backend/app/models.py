from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, Boolean, Numeric, JSON, Index
)

from sqlalchemy.orm import relationship
from datetime import datetime
from .extensions import Base

class Assignment(Base):
    __tablename__ = "assignments"

    id = Column(String, primary_key=True)  # uuid string
    filename = Column(String, nullable=False)
    storage_uri = Column(String, nullable=False)  # file://... or s3://... later
    status = Column(String, nullable=False, default="uploaded")  # uploaded|grading|graded|failed

    suggested_grade = Column(Integer, nullable=True)  # 0-100 (example)
    feedback = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String)
    role = Column(String, nullable=False)  # student|teacher|admin
    created_at = Column(DateTime, default=datetime.utcnow)

class Course(Base):
    __tablename__ = "courses"
    id = Column(Integer, primary_key=True)
    code = Column(String, nullable=False)
    title = Column(String, nullable=False)

class Enrollment(Base):
    __tablename__ = "enrollments"
    id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    role = Column(String, nullable=False)  # student|teacher

    course = relationship("Course")
    user = relationship("User")

Index("ix_enroll_course_user", Enrollment.course_id, Enrollment.user_id, unique=True)

# class Assignment(Base):
#     __tablename__ = "assignments"
#     id = Column(Integer, primary_key=True)
#     course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
#     title = Column(String, nullable=False)
#     description = Column(Text)
#     modality = Column(String, nullable=False)  # text|code|video
#     rubric = Column(JSON, nullable=False, default=list)  # list of criteria
#     created_at = Column(DateTime, default=datetime.utcnow)

#     course = relationship("Course")

class Submission(Base):
    __tablename__ = "submissions"
    id = Column(Integer, primary_key=True)
    assignment_id = Column(Integer, ForeignKey("assignments.id"), nullable=False)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(String, default="queued")  # queued|grading|graded|needs_review|error
    created_at = Column(DateTime, default=datetime.utcnow)

    final_score = Column(Numeric(5,2))
    final_feedback = Column(Text)

    assignment = relationship("Assignment")
    student = relationship("User")
    artifacts = relationship("SubmissionArtifact", back_populates="submission")

class SubmissionArtifact(Base):
    __tablename__ = "submission_artifacts"
    id = Column(Integer, primary_key=True)
    submission_id = Column(Integer, ForeignKey("submissions.id"), nullable=False)
    kind = Column(String, nullable=False)  # pdf|txt|ipynb|zip|mp4|png|jpg
    s3_key = Column(String, nullable=False)
    sha256 = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    submission = relationship("Submission", back_populates="artifacts")

class AIScore(Base):
    __tablename__ = "ai_scores"
    id = Column(Integer, primary_key=True)
    submission_id = Column(Integer, ForeignKey("submissions.id"), nullable=False)
    criterion = Column(String, nullable=False)
    score = Column(Numeric(5,2), nullable=False)
    confidence = Column(Numeric(3,2), nullable=False)
    rationale = Column(Text, nullable=False)
    evidence = Column(JSON, nullable=True)
    model = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True)
    actor_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String, nullable=False)  # VIEW_SUBMISSION, OVERRIDE_GRADE, etc.
    target_type = Column(String, nullable=False)  # Submission, Assignment, User
    target_id = Column(Integer, nullable=False)
    # metadata = Column(JSON, default=dict)
    event_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
