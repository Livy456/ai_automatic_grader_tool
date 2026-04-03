"""public autograder: nullable course_id/student_id + grader context + submitter_ip

Revision ID: p0q1r2s3t4u5
Revises: a9b8c7d6e5f4

Standalone autograder writes to assignments (course_id NULL), submissions, submission_artifacts, ai_scores.

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "p0q1r2s3t4u5"
down_revision: Union[str, None] = "a9b8c7d6e5f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "assignments",
        "course_id",
        existing_type=sa.Integer(),
        nullable=True,
    )
    op.add_column("assignments", sa.Column("grader_rubric_text", sa.Text(), nullable=True))
    op.add_column("assignments", sa.Column("grader_answer_key_text", sa.Text(), nullable=True))
    op.add_column("assignments", sa.Column("grader_instructions", sa.Text(), nullable=True))
    op.alter_column(
        "submissions",
        "student_id",
        existing_type=sa.Integer(),
        nullable=True,
    )
    op.add_column("submissions", sa.Column("submitter_ip", sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column("submissions", "submitter_ip")
    op.alter_column(
        "submissions",
        "student_id",
        existing_type=sa.Integer(),
        nullable=False,
    )
    op.drop_column("assignments", "grader_instructions")
    op.drop_column("assignments", "grader_answer_key_text")
    op.drop_column("assignments", "grader_rubric_text")
    op.alter_column(
        "assignments",
        "course_id",
        existing_type=sa.Integer(),
        nullable=False,
    )
