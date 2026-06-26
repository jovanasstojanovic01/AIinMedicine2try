
from re import search
from flask import Blueprint, request
from marshmallow import ValidationError
from app.extensions import db
from app.models.db_models import Pacijent, Pregled  
from app.models.schemas import patient_schema, patients_schema  
from app.ml.ml_service import ml_service
from app.utils.responses import ok, created, error, not_found
from datetime import datetime

bp = Blueprint("patients", __name__, url_prefix="/api/patients")

@bp.get("/<int:patient_id>/predict-progression")
def evaluate_progression(patient_id):
    eye = request.args.get("eye", "OD").upper()
    if eye not in ["OD", "OS"]:
        return error("Parametar 'eye' mora biti 'OD' ili 'OS'.", 400)

    pacijent = Pacijent.query.get(patient_id)
    if pacijent is None:
        return not_found("Pacijent nije pronađen.")

    
    pregledi = Pregled.query.filter_by(patient_id=patient_id).order_by(Pregled.visit_number).all()

    if len(pregledi) == 0:
        return error("Pacijent nema evidentiranih pregleda u bazi podataka.", 400)

    
    sequence_history = []
    for p in pregledi:
        
        vcdr_val = 0.0
        if p.multimedija:
            vcdr_val = p.multimedija.od_vcdr if eye == "OD" else p.multimedija.os_vcdr

        
        iop = p.od_iop if eye == "OD" else p.os_iop
        md = p.od_md if eye == "OD" else p.os_md
        oct_mean = p.od_oct_mean if eye == "OD" else p.os_oct_mean
        
        sequence_history.append([
            float(iop or 0.0),
            float(md or 0.0),
            float(oct_mean or 0.0),
            float(vcdr_val or 0.0),
            float(pacijent.cct or 540.0)
        ])

    try:
        prediction = ml_service.predict_progression(sequence_history)
        
        
        poslednji_pregled = pregledi[-1]
        if eye == "OD":
            poslednji_pregled.od_progression_status = int(prediction.get("progression", 0))
        else:
            poslednji_pregled.os_progression_status = int(prediction.get("progression", 0))
        db.session.commit()

        return ok(prediction, f"Predikcija progresije za {eye} oko uspešno izvršena.")
    except Exception as e:
        db.session.rollback()
        return error(f"Greška tokom predikcije: {str(e)}", 500)


@bp.get("")
def list_patients():
    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 20, type=int), 100)
    search = request.args.get("search", "").strip()
    birth_date_str = request.args.get("birth_date", "").strip()

    q = Pacijent.query
    q = Pacijent.query
    if search:
        
        like = f"%{search}%"
        
        
        ime_prezime = db.func.concat(Pacijent.first_name, ' ', Pacijent.last_name)
        
        prezime_ime = db.func.concat(Pacijent.last_name, ' ', Pacijent.first_name)
        
        
        q = q.filter(
            ime_prezime.ilike(like) | 
            prezime_ime.ilike(like) |
            Pacijent.first_name.ilike(like) |
            Pacijent.last_name.ilike(like)
        )
    if birth_date_str:
        try:
            
            stvarno_vreme = datetime.strptime(birth_date_str, "%Y-%m-%d").date()
            q = q.filter(Pacijent.birth_date == stvarno_vreme)
        except ValueError:
            
            
            pass

    pagination = q.order_by(Pacijent.last_name, Pacijent.first_name).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    serialized_patients = patients_schema.dump(pagination.items)

    return ok({
        "patients": serialized_patients,
        "total": pagination.total,
        "pages": pagination.pages,
        "page": page,
    })


@bp.get("/<int:patient_id>")
def get_patient(patient_id):
    p = Pacijent.query.get(patient_id)
    if p is None:
        return not_found("Pacijent nije pronađen.")
    
    return ok(patient_schema.dump(p))


@bp.put("/<int:patient_id>")
def update_patient(patient_id):
    p = Pacijent.query.get(patient_id)
    if p is None:
        return not_found("Pacijent nije pronađen.")

    json_data = request.get_json(silent=True)
    if not json_data:
        return error("Body zahteva mora biti JSON format.", 400)

    try:
        
        p = patient_schema.load(json_data, instance=p, session=db.session, partial=True)
    except ValidationError as exc:
        return error("Validacija neuspešna.", 422, exc.messages)

    db.session.commit()
    return ok(patient_schema.dump(p), "Podaci o pacijentu uspešno ažurirani.")

@bp.post("")
def create_patient():
    json_data = request.get_json(silent=True)
    if not json_data:
        return error("Zahtev mora sadržati JSON body.", 400)

    try:
        
        novi_pacijent = patient_schema.load(json_data, session=db.session)
    except ValidationError as exc:
        return error("Validacija polja neuspešna.", 422, exc.messages)

    try:
        db.session.add(novi_pacijent)
        db.session.commit() 

        return created(patient_schema.dump(novi_pacijent), "Profil pacijenta uspešno kreiran.")
    except Exception as e:
        db.session.rollback()
        return error(f"Greška: {str(e)}", 500)
@bp.delete("/<int:patient_id>")
def delete_patient(patient_id):
    p = Pacijent.query.get(patient_id)
    if p is None:
        return not_found("Pacijent nije pronađen.")

    db.session.delete(p)
    db.session.commit()
    return ok(message="Profil pacijenta i svi njegovi pregledi su uspešno obrisani.")