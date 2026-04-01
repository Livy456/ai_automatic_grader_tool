"""submission direct upload lifecycle columns

Revision ID: d4e5f6a7b8c9
Revises: 0a9fbf797fc1
Create Date: 2026-03-29

Adds columns for presigned S3 upload flow and idempotent grading dispatch.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, None] = "0a9fbf797fc1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # submissions may already exist outside this migration chain; use IF NOT EXISTS (Postgres).
    op.execute(
        sa.text(
            """
            ALTER TABLE submissions
            ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP;
            UPDATE submissions SET updated_at = created_at WHERE updated_at IS NULL;
            """
        )
    )
    op.execute(
        sa.text(
            """
            ALTER TABLE submissions
            ADD COLUMN IF NOT EXISTS grading_dispatch_at TIMESTAMP;
            """
        )
    )
    op.execute(
        sa.text(
            """
            ALTER TABLE submissions
            ADD COLUMN IF NOT EXISTS grading_celery_task_id VARCHAR(128);
            """
        )
    )


def downgrade() -> None:
    op.execute(sa.text("ALTER TABLE submissions DROP COLUMN IF EXISTS grading_celery_task_id"))
    op.execute(sa.text("ALTER TABLE submissions DROP COLUMN IF EXISTS grading_dispatch_at"))
    op.execute(sa.text("ALTER TABLE submissions DROP COLUMN IF EXISTS updated_at"))
