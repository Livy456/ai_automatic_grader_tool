"""add grading_instructions to standalone_submissions

Revision ID: a9b8c7d6e5f4
Revises: f1a2b3c4d5e6

Optional instructor prompt used with rubric / sample response in the Ollama grading pipeline.

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a9b8c7d6e5f4"
down_revision: Union[str, None] = "f1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "standalone_submissions",
        sa.Column("grading_instructions", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("standalone_submissions", "grading_instructions")
