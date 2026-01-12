from flask import Flask
from flask_cors import CORS
from .config import Config
# from extensions import init_db, Base, engine

# from app import extensions
# from app.extensions import Base
from app.extensions import Base, init_db

from .auth import bp as auth_bp, init_oauth
from .tasks import init_celery
from .routes.health import bp as health_bp
# from .routes.assignments import bp as assignments_bp # OLD VERSION OF THE ASSIGNMENTS
from .routes.submissions import bp as submissions_bp
from .routes.admin import bp as admin_bp
from .routes_assignments import bp as assignments_bp



def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config_obj = Config()  # for celery task access
    CORS(app, supports_credentials=True)

    # init_db(app.config["DATABASE_URL"])
    # Base.metadata.create_all(bind=engine)
    
    # extensions.init_db(app.config["DATABASE_URL"])
    # Base.metadata.create_all(bind=extensions.engine)

    print("DATABASE_URL =", app.config.get("DATABASE_URL"))
    print("DATABASE_URL_ACTUAL: ", app.config["DATABASE_URL"])
    engine = init_db(app.config["DATABASE_URL"])
    Base.metadata.create_all(bind=engine)

    init_oauth(app)
    init_celery(app)

    app.register_blueprint(auth_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(assignments_bp)
    app.register_blueprint(submissions_bp)
    app.register_blueprint(admin_bp)
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
