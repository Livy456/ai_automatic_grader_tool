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
    # Null when the assignment is created by the public standalone autograder (no course).
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    modality = Column(String, nullable=False)
    rubric = Column(JSON, nullable=False)
    created_at = Column(DateTime)
    due_date = Column(DateTime, nullable=True)
    # Optional text context for public autograder rows (course_id IS NULL); used by grade_submission.
    grader_rubric_text = Column(Text, nullable=True)
    grader_answer_key_text = Column(Text, nullable=True)
    grader_instructions = Column(Text, nullable=True)

    course = relationship("Course")
    attachments = relationship("AssignmentAttachment", back_populates="assignment")


class AssignmentAttachment(Base):
    """Teacher-uploaded rubric files / answer keys for a course Assignment (integer id)."""

    __tablename__ = "assignment_attachments"

    id = Column(Integer, primary_key=True)
    assignment_id = Column(Integer, ForeignKey("assignments.id"), nullable=False)
    kind = Column(String(32), nullable=False)  # rubric | answer_key
    s3_key = Column(String(1024), nullable=False)
    filename = Column(String(512), nullable=False)
    uploaded_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    assignment = relationship("Assignment", back_populates="attachments")
    uploaded_by = relationship("User")


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String)
    role = Column(String, nullable=False)  # student|teacher|admin
    created_at = Column(DateTime, default=datetime.utcnow)
    # Local login (OAuth users leave password_hash null)
    password_hash = Column(String(255), nullable=True)
    institution_domain = Column(String(255), nullable=True)
    first_login_at = Column(DateTime, nullable=True)
    last_login_at = Column(DateTime, nullable=True)


class IssuedJwt(Base):
    """Server-side record for each issued access token (jti allowlist + logout/revocation)."""
    __tablename__ = "issued_jwts"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    jti = Column(String(64), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class RefreshToken(Base):
    """Opaque refresh token (hash stored); raw value is placed in an HttpOnly cookie only."""

    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token_hash = Column(String(64), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Course(Base):
    __tablename__ = "courses"
    id = Column(Integer, primary_key=True)
    code = Column(String, nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)

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
    """
    Lifecycle (status):
      uploading → uploaded → queued → grading → graded | needs_review | error
    Direct S3 flow: create as uploading; after browser PUTs to S3, finalize sets uploaded,
    then atomically queued + single Celery enqueue (grading_dispatch_at set once).
    """

    __tablename__ = "submissions"
    id = Column(Integer, primary_key=True)
    assignment_id = Column(Integer, ForeignKey("assignments.id"), nullable=False)
    # Null for anonymous public autograder uploads; set when a JWT is present at upload time.
    student_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    status = Column(String(32), nullable=False, default="uploading")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Set once when grade_submission.delay succeeds — idempotent finalize / enqueue.
    grading_dispatch_at = Column(DateTime, nullable=True)
    grading_celery_task_id = Column(String(128), nullable=True)

    final_score = Column(Numeric(5, 2))
    final_feedback = Column(Text)
    # Best-effort client IP for anonymous autograder rate limiting / mutation checks.
    submitter_ip = Column(String(64), nullable=True)

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


class StandaloneSubmission(Base):
    """
    Standalone autograder submission — not tied to a course or assignment.
    """

    __tablename__ = "standalone_submissions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(512), nullable=False, default="Untitled")
    status = Column(String(32), nullable=False, default="uploading")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    grading_dispatch_at = Column(DateTime, nullable=True)
    grading_celery_task_id = Column(String(128), nullable=True)

    final_score = Column(Numeric(5, 2), nullable=True)
    final_feedback = Column(Text, nullable=True)

    rubric_text = Column(Text, nullable=True)
    answer_key_text = Column(Text, nullable=True)
    # Optional free-text prompt (focus, learning goals) combined with rubric / sample in the grader.
    grading_instructions = Column(Text, nullable=True)

    user = relationship("User")
    artifacts = relationship(
        "StandaloneArtifact", back_populates="submission", cascade="all, delete-orphan"
    )
    scores = relationship(
        "StandaloneAIScore", back_populates="submission", cascade="all, delete-orphan"
    )


class StandaloneArtifact(Base):
    __tablename__ = "standalone_artifacts"

    id = Column(Integer, primary_key=True)
    submission_id = Column(Integer, ForeignKey("standalone_submissions.id"), nullable=False)
    kind = Column(String(32), nullable=False)
    s3_key = Column(String(1024), nullable=False)
    filename = Column(String(512), nullable=False)
    sha256 = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    submission = relationship("StandaloneSubmission", back_populates="artifacts")


class StandaloneAIScore(Base):
    __tablename__ = "standalone_ai_scores"

    id = Column(Integer, primary_key=True)
    submission_id = Column(Integer, ForeignKey("standalone_submissions.id"), nullable=False)
    criterion = Column(String, nullable=False)
    score = Column(Numeric(5, 2), nullable=False)
    confidence = Column(Numeric(3, 2), nullable=False)
    rationale = Column(Text, nullable=False)
    evidence = Column(JSON, nullable=True)
    model = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    submission = relationship("StandaloneSubmission", back_populates="scores")
