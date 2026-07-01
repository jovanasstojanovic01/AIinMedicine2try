
from app.extensions import db

from app.models.db_models import Pacijent, Pregled, PregledMultimedija
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from marshmallow import fields




class PregledMultimedijaSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = PregledMultimedija
        load_instance = True
        sqla_session = db.session
        include_fk = True  





class PacijentSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Pacijent
        load_instance = True
        sqla_session = db.session

    
    birth_date = fields.Date(format="%Y-%m-%d", required=True)





class PregledSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Pregled
        load_instance = True
        sqla_session = db.session
        include_fk = True

    
    
    exam_date = fields.Date(format="%Y-%m-%d", required=True)

    
    pacijent = fields.Nested(
        PacijentSchema, 
        only=("patient_id", "first_name", "last_name", "gender", "glaucoma_category")
    )
    
    
    od_multimedija = fields.Nested(PregledMultimedijaSchema, allow_none=True)
    os_multimedija = fields.Nested(PregledMultimedijaSchema, allow_none=True)





patient_schema = PacijentSchema()
visit_schema = PregledSchema()

patients_schema = PacijentSchema(many=True)
visits_schema = PregledSchema(many=True)