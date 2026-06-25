
import os
from flask import Blueprint, request,current_app
from marshmallow import ValidationError
from app.extensions import db
from app.models.db_models import Pacijent, Pregled, PregledMultimedija  
from app.models.schemas import visit_schema, visits_schema          
from app.ml.ml_service import ml_service
from app.utils.responses import ok, created, error, not_found
from datetime import datetime

bp = Blueprint("visits", __name__, url_prefix="/api/visits")




@bp.get("")
def list_all_visits():
    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 10, type=int), 100)
    
    pagination = Pregled.query.order_by(Pregled.exam_date.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    
    serialized_visits = visits_schema.dump(pagination.items)
        
    return ok({
        "visits": serialized_visits,
        "total": pagination.total,
        "pages": pagination.pages,
        "page": page
    })





@bp.get("/<int:exam_id>")
def get_visit(exam_id):
    v = Pregled.query.get(exam_id)
    if v is None:
        return not_found("Pregled nije pronađen.")
    
    return ok(visit_schema.dump(v))



def sacuvaj_i_analiziraj_sliku(img_file):
    if not img_file or img_file.filename == '':
        return None, None, 0.0

    filename = img_file.filename
    
    img_path = os.path.join(current_app.config['IMAGES_FOLDER'], filename)
    
    
    img_bytes = img_file.read()
    img_file.seek(0)
    img_file.save(img_path)

    
    ml_res = ml_service.predict_glaucoma_segmentation(img_bytes)
    
    
    ime_bez_ekstenzije, _ = os.path.splitext(filename)
    mask_filename = f"{ime_bez_ekstenzije}_mask.png"
    mask_path = os.path.join(current_app.config['MASKS_FOLDER'], mask_filename)

    
    
    if "mask_bytes" in ml_res:
        with open(mask_path, "wb") as f:
            f.write(ml_res["mask_bytes"])
    
    return filename, ml_res["vcdr"], ml_res.get("status", "")

@bp.post("/initial")
def create_initial_visit():
    form = request.form
    files = request.files

    patient_id = form.get("patient_id", type=int)
    if not patient_id or Pacijent.query.get(patient_id):
        return error("Nevalidan ili postojeći 'patient_id'.", 400)

    try:
        # # 1. Kreiranje pacijenta
        # birth_date_str = form.get("birth_date", "1970-01-01")
        # novi_pacijent = Pacijent(
        #     patient_id=patient_id,
        #     first_name=form.get("first_name"),
        #     last_name=form.get("last_name"),
        #     gender=form.get("gender"),
        #     birth_date=datetime.strptime(birth_date_str, "%Y-%m-%d").date(),
        #     cct=form.get("cct", type=float),
        #     glaucoma_category=form.get("glaucoma_category")
        # )
        # db.session.add(novi_pacijent)
        # db.session.flush()

        # 2. Kreiranje pregleda
        exam_date_str = form.get("exam_date", datetime.utcnow().strftime("%Y-%m-%d"))
        pregled = Pregled(
            patient_id=patient_id,
            visit_number=0,
            exam_date=datetime.strptime(exam_date_str, "%Y-%m-%d").date(),
            interval_godina=0.0,
            od_iop=form.get("od_iop", type=float), os_iop=form.get("os_iop", type=float),
            physician_comment=form.get("physician_comment"), therapy=form.get("therapy")
        )
        db.session.add(pregled)
        db.session.flush()

        # 3. Obrada i čuvanje slika na disk + unos u bazu
        filename_od, vcdr_od, status_od = sacuvaj_i_analiziraj_sliku(files.get("image_OD"))
        filename_os, vcdr_os, status_os = sacuvaj_i_analiziraj_sliku(files.get("image_OS"))

        multimedija = PregledMultimedija(
            exam_id=pregled.exam_id,
            od_image=filename_od,
            os_image=filename_os,
            od_vcdr=vcdr_od, od_hcdr=vcdr_od, od_acdr=vcdr_od, od_rim_area_pixels=4500.0,
            os_vcdr=vcdr_os, os_hcdr=vcdr_os, os_acdr=vcdr_os, os_rim_area_pixels=4500.0
        )
        
        pregled.ai_predlog_terapije = f"OD Status: {status_od} | OS Status: {status_os}"
        
        db.session.add(multimedija)
        db.session.commit()

        return created(visit_schema.dump(pregled), "Inicijalni karton i slike uspešno sačuvani.")

    except Exception as e:
        db.session.rollback()
        return error(f"Greška na serveru: {str(e)}", 500)

# ─────────────────────────────────────────────────────────────────────────────
# POST /control (Usklađeno sa čuvanjem fajlova)
# ─────────────────────────────────────────────────────────────────────────────
@bp.post("/control")
def create_control_visit():
    form = request.form
    files = request.files

    patient_id = form.get("patient_id", type=int)
    pacijent = Pacijent.query.get(patient_id)
    if not pacijent:
        return not_found("Pacijent nije pronađen.")

    try:
        poslednji_pregled = Pregled.query.filter_by(patient_id=patient_id).order_by(Pregled.visit_number.desc()).first()
        sledeci_broj = (poslednji_pregled.visit_number + 1) if poslednji_pregled else 1

        pregled = Pregled(
            patient_id=patient_id,
            visit_number=sledeci_broj,
            exam_date=datetime.strptime(form.get("exam_date", datetime.utcnow().strftime("%Y-%m-%d")), "%Y-%m-%d").date(),
            interval_godina=form.get("interval_godina", 1.0, type=float),
            od_iop=form.get("od_iop", type=float), os_iop=form.get("os_iop", type=float),
            physician_comment=form.get("physician_comment"), therapy=form.get("therapy")
        )
        db.session.add(pregled)
        db.session.flush()

        filename_od, vcdr_od, status_od = sacuvaj_i_analiziraj_sliku(files.get("image_OD"))
        filename_os, vcdr_os, status_os = sacuvaj_i_analiziraj_sliku(files.get("image_OS"))

        multimedija = PregledMultimedija(
            exam_id=pregled.exam_id,
            od_image=filename_od, os_image=filename_os,
            od_vcdr=vcdr_od, od_hcdr=vcdr_od, od_acdr=vcdr_od, od_rim_area_pixels=4500.0,
            os_vcdr=vcdr_os, os_hcdr=vcdr_os, os_acdr=vcdr_os, os_rim_area_pixels=4500.0
        )
        db.session.add(multimedija)
        db.session.commit()

        return created(visit_schema.dump(pregled), f"Kontrolni pregled br. {sledeci_broj} sačuvan sa medijima.")
    except Exception as e:
        db.session.rollback()
        return error(str(e), 500)





@bp.put("/<int:exam_id>")
def update_visit(exam_id):
    v = Pregled.query.get(exam_id)
    if v is None:
        return not_found("Pregled nije pronađen.")

    json_data = request.get_json(silent=True)
    if not json_data:
        return error("Zahtev mora sadržati JSON body.", 400)

    try:
        
        v = visit_schema.load(json_data, instance=v, session=db.session, partial=True)
    except ValidationError as exc:
        return error("Validacija polja neuspešna.", 422, exc.messages)

    db.session.commit()
    return ok(visit_schema.dump(v), "Podaci o pregledu uspešno ažurirani.")





@bp.delete("/<int:exam_id>")
def delete_visit(exam_id):
    v = Pregled.query.get(exam_id)
    if v is None:
        return not_found("Pregled nije pronađen.")

    
    db.session.delete(v)
    db.session.commit()
    return ok(message="Pregled i prateći podaci o slikama su uspešno obrisani.")