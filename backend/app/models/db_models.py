
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
    od_vf_matrix = db.Column(db.Text, nullable=True)
    od_vf_file = db.Column(db.String(255), nullable=True)
    # od_md                 = db.Column(db.Float)
    # od_oct_mean           = db.Column(db.Float)
    # od_oct_s              = db.Column(db.Float)
    # od_oct_n              = db.Column(db.Float)
    # od_oct_i              = db.Column(db.Float)
    # od_oct_t              = db.Column(db.Float)
    # od_progression_status = db.Column(db.Integer, default=0)

    
    os_iop                = db.Column(db.Float)
    os_vf_matrix = db.Column(db.Text, nullable=True)
    os_vf_file = db.Column(db.String(255), nullable=True)
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
    multimedija = db.relationship("PregledMultimedija", back_populates="pregled", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Pregled {self.exam_id} pacijent={self.patient_id} poseta={self.visit_number}>"





class PregledMultimedija(db.Model):
    __tablename__ = "table_multimedia"

    multimedia_id      = db.Column(db.Integer, primary_key=True, autoincrement=True)
    exam_id            = db.Column(db.Integer, db.ForeignKey("table_exams.exam_id"), nullable=False, unique=True)

    
    od_image    = db.Column(db.String(255))
    
    os_image    = db.Column(db.String(255))
    od_vcdr    = db.Column(db.Float)
    od_hcdr    = db.Column(db.Float)
    od_acdr    = db.Column(db.Float)
    od_rim_area_pixels = db.Column(db.Float)

    os_vcdr            = db.Column(db.Float)
    os_hcdr            = db.Column(db.Float)
    os_acdr            = db.Column(db.Float)
    os_rim_area_pixels = db.Column(db.Float)

    pregled = db.relationship("Pregled", back_populates="multimedija")