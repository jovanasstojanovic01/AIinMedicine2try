
from datetime import datetime
from app.extensions import db




class Pacijent(db.Model):
    __tablename__ = "table_patients"

    patient_id        = db.Column(db.Integer, primary_key=True, autoincrement=True)
    first_name        = db.Column(db.String(100), nullable=False)
    last_name         = db.Column(db.String(100), nullable=False)
    gender            = db.Column(db.Enum("M", "F"), nullable=False)
    birth_date        = db.Column(db.Date, nullable=False)
    cct               = db.Column(db.Float, nullable=False)
    glaucoma_category = db.Column(db.Enum("None", "ACG", "OAG", name="glaucoma_categories"), nullable=False)

    
    pregledi = db.relationship("Pregled", back_populates="pacijent", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Pacijent {self.patient_id}: {self.first_name} {self.last_name}>"





class Pregled(db.Model):
    __tablename__ = "table_exams"

    exam_id              = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_id           = db.Column(db.Integer, db.ForeignKey("table_patients.patient_id"), nullable=False)
    visit_number         = db.Column(db.Integer, nullable=False)
    exam_date            = db.Column(db.Date, nullable=False)
    #interval_godina      = db.Column(db.Float, nullable=True) 

    
    od_iop                = db.Column(db.Float)
    od_diagnosis=db.Column(db.Enum("Glaucoma Suspect / Positive","Healthy"), nullable=True)
    od_vf_matrix = db.Column(db.Text, nullable=True)
    od_vf_file = db.Column(db.String(255), nullable=True)
    od_multimedia_id = db.Column(db.Integer, db.ForeignKey("table_multimedia.multimedia_id"), nullable=True)
    od_next_vf_mean_pred = db.Column(db.Float, nullable=True)
    # od_md                 = db.Column(db.Float)
    # od_oct_mean           = db.Column(db.Float)
    # od_oct_s              = db.Column(db.Float)
    # od_oct_n              = db.Column(db.Float)
    # od_oct_i              = db.Column(db.Float)
    # od_oct_t              = db.Column(db.Float)
    # od_progression_status = db.Column(db.Integer, default=0)

    
    os_iop                = db.Column(db.Float)
    os_diagnosis=db.Column(db.Enum("Glaucoma Suspect / Positive","Healthy"), nullable=True)
    os_vf_matrix = db.Column(db.Text, nullable=True)
    os_vf_file = db.Column(db.String(255), nullable=True)
    os_multimedia_id = db.Column(db.Integer, db.ForeignKey("table_multimedia.multimedia_id"), nullable=True)
    os_next_vf_mean_pred = db.Column(db.Float, nullable=True)
    # os_md                 = db.Column(db.Float)
    # os_oct_mean           = db.Column(db.Float)
    # os_oct_s              = db.Column(db.Float)
    # os_oct_n              = db.Column(db.Float)
    # os_oct_i              = db.Column(db.Float)
    # os_oct_t              = db.Column(db.Float)
    # os_progression_status = db.Column(db.Integer, default=0)

    
    physician_comment    = db.Column(db.Text)
    therapy              = db.Column(db.Text)
    #ai_predlog_terapije  = db.Column(db.Text, nullable=True)

    
    pacijent    = db.relationship("Pacijent", back_populates="pregledi")
    od_multimedija = db.relationship("PregledMultimedija", foreign_keys=[od_multimedia_id], cascade="all, delete-orphan", single_parent=True)
    os_multimedija = db.relationship("PregledMultimedija", foreign_keys=[os_multimedia_id], cascade="all, delete-orphan", single_parent=True)
    def __repr__(self):
        return f"<Pregled {self.exam_id} pacijent={self.patient_id} poseta={self.visit_number}>"





class PregledMultimedija(db.Model):
    __tablename__ = "table_multimedia"

    multimedia_id      = db.Column(db.Integer, primary_key=True, autoincrement=True)
    image_path      = db.Column(db.String(255), nullable=True)

    vcdr            = db.Column(db.Float)
    hcdr            = db.Column(db.Float)
    acdr            = db.Column(db.Float)
    rim_area_pixels = db.Column(db.Float)