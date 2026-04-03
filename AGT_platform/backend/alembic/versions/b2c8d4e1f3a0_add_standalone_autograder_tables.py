"""add standalone autograder tables

Revision ID: b2c8d4e1f3a0
Revises: e2f3a4b5c6d7

Linear tip after course/assignment migrations. ``users`` must exist (f8e7… chain).

If your database has ``alembic_version.version_num = 'ad05c0c8d223'`` (from a Docker-only
autogenerate) and that revision file is not in this repo, run **once** (dev DB; backup if unsure):

  UPDATE alembic_version SET version_num = 'e2f3a4b5c6d7' WHERE version_num = 'ad05c0c8d223';

then ``alembic upgrade head`` to apply this revision.

Create Date: 2026-03-29

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b2c8d4e1f3a0"
down_revision: Union[str, None] = "e2f3a4b5c6d7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "standalone_submissions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("grading_dispatch_at", sa.DateTime(), nullable=True),
        sa.Column("grading_celery_task_id", sa.String(length=128), nullable=True),
        sa.Column("final_score", sa.Numeric(5, 2), nullable=True),
        sa.Column("final_feedback", sa.Text(), nullable=True),
        sa.Column("rubric_text", sa.Text(), nullable=True),
        sa.Column("answer_key_text", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "standalone_artifacts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("submission_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("s3_key", sa.String(length=1024), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("sha256", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["submission_id"], ["standalone_submissions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "standalone_ai_scores",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("submission_id", sa.Integer(), nullable=False),
        sa.Column("criterion", sa.String(), nullable=False),
        sa.Column("score", sa.Numeric(5, 2), nullable=False),
        sa.Column("confidence", sa.Numeric(3, 2), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=True),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["submission_id"], ["standalone_submissions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("standalone_ai_scores")
    op.drop_table("standalone_artifacts")
    op.drop_table("standalone_submissions")
