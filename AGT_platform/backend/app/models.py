from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, ForeignKey, Boolean, Numeric, JSON, Index, UUID
)
from sqlalchemy.orm import relationship, DeclarativeBase
from datetime import datetime
import uuid
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

class Base(DeclarativeBase):
    pass

def now():
    return datetime.now()

class AssignmentUpload(Base):
    __tablename__ = "assignment_uploads"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(512), nullable=False)
    storage_uri = Column(Text, nullable=False)
    status = Column(String(32), nullable=False, default="uploaded")  # uploaded|grading|graded|error
    suggested_grade = Column(Float, nullable=True)
    feedback = Column(Text, nullable=True)

    created_at = Column(DateTime, default=now)
    updated_at = Column(DateTime, default=now, onupdate=now)

class Assignment(Base):
    __tablename__ = "assignments"

    id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text)
    modality = Column(String, nullable=False)
    rubric = Column(JSON, nullable=False)
    created_at = Column(DateTime)

    course = relationship("Course")

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

class Submission(Base):
    __tablename__ = "submissions"
    id = Column(Integer, primary_key=True)
    assignment_id = Column(Integer, ForeignKey("assignments.id"), nullable=False)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(String, default="queued")  # queued|grading|graded|needs_review|error
    created_at = Column(DateTime, default=datetime.now)

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
