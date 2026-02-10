Make sure you are in the backend directory

1. Run the following command in the terminal to install packages used for the backend:

>> pip install -r requirements.txt

2. If you are using the autograder and not MIT-affiliated coruse, consult your school's IT docs for “OpenID Connect discovery URL” / “Issuer” / “OIDC”. Then replace the value for OIDC_DISCOVERY_URL with your school's appropriate OpenID Connect URL

3. Install alembic to have database migrations

>> python -m pip install alembic

4. Initialize alembic in the backend

>> alembic init alembic

5. Generate the migration (for wsl)

>> export DATABASE_URL="postgresql://dev:dev@localhost:5432/ai_grader"
>> alembic revision --autogenerate -m "create assignment_uploads"
>> alembic upgrade head

6. Run the following command in the terminal to run the backend:

>> python -m app.main

7. Access the backend information locally from this link:

>> INSERT LINK EVENTUALLY

When migrating new tables to the database:

python -m alembic current
python -m alembic revision --autogenerate -m "create assignment_uploads"
python -m alembic upgrade head

alembic revision --autogenerate -m "create assignment_uploads"
