"""add course description and assignment due_date

Revision ID: e2f3a4b5c6d7
Revises: d4e5f6a7b8c9
Create Date: 2026-03-29

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e2f3a4b5c6d7"
down_revision: Union[str, None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("courses", sa.Column("description", sa.Text(), nullable=True))
    op.add_column("assignments", sa.Column("due_date", sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column("assignments", "due_date")
    op.drop_column("courses", "description")
