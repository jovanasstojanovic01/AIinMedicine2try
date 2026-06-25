
import os
from flask import Flask
from app.config import Config
from app.extensions import db

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    
    db.init_app(app)

    
    

    
   
    os.makedirs(app.config['IMAGES_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MASKS_FOLDER'], exist_ok=True)
    with app.app_context():
        from app.routes.patients import bp as patients_bp
        from app.routes.visits import bp as visits_bp
        from app.routes.media import bp as media_bp

        
        
        app.register_blueprint(patients_bp)
        app.register_blueprint(visits_bp)
        app.register_blueprint(media_bp)
        db.create_all()
    return app