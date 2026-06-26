
import os
from flask import Blueprint, request,current_app
from marshmallow import ValidationError
from app.extensions import db
from app.models.db_models import Pacijent, Pregled, PregledMultimedija  
from app.models.schemas import visit_schema, visits_schema          
from app.ml.ml_service import ml_service
from app.utils.media_helpers import generisi_jedinstveno_ime, get_mask_filename
from app.utils.responses import ok, created, error, not_found
from datetime import datetime

bp = Blueprint("visits", __name__, url_prefix="/api/visits")



@bp.post("/<int:exam_id>/predict-progression")
def evaluate_visit_progression(exam_id):
    """
    Pokreće XGBoost/LSTM model za predikciju progresije glaukoma u kontekstu konkretnog pregleda.
    Uzima u obzir istoriju pregleda pacijenta ZAKLJUČNO sa ovim pregledom.
    Sve se automatski upisuje u bazu za taj pregled.
    """
    eye = request.args.get("eye", "OD").upper()
    if eye not in ["OD", "OS"]:
        return error("Parametar 'eye' mora biti 'OD' ili 'OS'.", 400)

    
    trenutni_pregled = Pregled.query.get(exam_id)
    if not trenutni_pregled:
        return not_found("Pregled nije pronađen.")

    
    pacijent = trenutni_pregled.pacijent 

    
    pregledi_istorija = Pregled.query.filter(
        Pregled.patient_id == trenutni_pregled.patient_id,
        Pregled.visit_number <= trenutni_pregled.visit_number
    ).order_by(Pregled.visit_number).all()

    
    sequence_history = []
    for p in pregledi_istorija:
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
        
        
        progression_status = int(prediction.get("progression", 0))
        
        if eye == "OD":
            trenutni_pregled.od_progression_status = progression_status
        else:
            trenutni_pregled.os_progression_status = progression_status
            
        db.session.commit()

        
        return ok(
            prediction, 
            f"Predikcija progresije za {eye} oko uspešno izvršena i sačuvana u pregled ID: {exam_id}."
        )
        
    except Exception as e:
        db.session.rollback()
        return error(f"Greška tokom predikcije: {str(e)}", 500)
@bp.post("")
def create_visit():
    json_data = request.get_json(silent=True)
    if not json_data:
        return error("Zahtev mora sadržati JSON body.", 400)

    patient_id = json_data.get("patient_id")
    if not patient_id:
        return error("Polje 'patient_id' je obavezno.", 400)

    pacijent = Pacijent.query.get(patient_id)
    if not pacijent:
        return not_found(f"Pacijent sa ID-jem {patient_id} ne postoji.")

    try:
        
        poslednji_pregled = Pregled.query.filter_by(patient_id=patient_id)\
                                         .order_by(Pregled.visit_number.desc())\
                                         .first()
        
        sledeci_visit_number = (poslednji_pregled.visit_number + 1) if poslednji_pregled else 0

        
        danasnji_datum = datetime.utcnow().date().strftime("%Y-%m-%d")
        
        json_data["visit_number"] = sledeci_visit_number
        if not json_data.get("exam_date"):
            json_data["exam_date"] = danasnji_datum
        
        
        novi_pregled = visit_schema.load(json_data, session=db.session)
        
        

        db.session.add(novi_pregled)
        db.session.commit()

        return created(
             visit_schema.dump(novi_pregled), 
             f"Pregled uspešno otvoren (Poseta br. {sledeci_visit_number})."
         )

    except ValidationError as exc:
        return error("Validacija podataka neuspešna.", 422, exc.messages)
    except Exception as e:
        db.session.rollback()
        return error(f"Greška na serveru: {str(e)}", 500)

@bp.post("/<int:exam_id>/upload-images")
def upload_visit_images(exam_id):
    pregled = Pregled.query.get(exam_id)
    if not pregled:
        return not_found("Pregled nije pronađen.")

    files = request.files
    img_od = files.get("image_OD")
    img_os = files.get("image_OS")

    if not img_od and not img_os:
        return error("Morate poslati bar jednu sliku ('image_OD' ili 'image_OS').", 400)

    try:
        
        multimedija = PregledMultimedija.query.filter_by(exam_id=exam_id).first()
        if not multimedija:
            multimedija = PregledMultimedija(exam_id=exam_id)
            db.session.add(multimedija)

        
        if img_od and img_od.filename != '':
            img_bytes = img_od.read()
            ml_res_od = ml_service.predict_glaucoma_segmentation(img_bytes)
            
            unikatno_ime_od = generisi_jedinstveno_ime(img_od.filename)
            img_path_od = os.path.join(current_app.config['IMAGES_FOLDER'], unikatno_ime_od)
            img_od.seek(0)
            img_od.save(img_path_od)

            mask_name_od = get_mask_filename(unikatno_ime_od)
            if "mask_bytes" in ml_res_od:
                with open(os.path.join(current_app.config['MASKS_FOLDER'], mask_name_od), "wb") as f:
                    f.write(ml_res_od["mask_bytes"])

            multimedija.od_image = unikatno_ime_od
            multimedija.od_vcdr = ml_res_od["vcdr"]
            multimedija.od_hcdr = ml_res_od["hcdr"]
            multimedija.od_acdr = ml_res_od["acdr"]
            multimedija.od_rim_area_pixels = float(ml_res_od["rim_area"])

        
        if img_os and img_os.filename != '':
            img_bytes = img_os.read()
            ml_res_os = ml_service.predict_glaucoma_segmentation(img_bytes)
            
            unikatno_ime_os = generisi_jedinstveno_ime(img_os.filename)
            img_path_os = os.path.join(current_app.config['IMAGES_FOLDER'], unikatno_ime_os)
            img_os.seek(0)
            img_os.save(img_path_os)

            mask_name_os = get_mask_filename(unikatno_ime_os)
            if "mask_bytes" in ml_res_os:
                with open(os.path.join(current_app.config['MASKS_FOLDER'], mask_name_os), "wb") as f:
                    f.write(ml_res_os["mask_bytes"])

            multimedija.os_image = unikatno_ime_os
            multimedija.os_vcdr = ml_res_os["vcdr"]
            multimedija.os_hcdr = ml_res_os["hcdr"]
            multimedija.os_acdr = ml_res_os["acdr"]
            multimedija.os_rim_area_pixels = float(ml_res_os["rim_area"])

        db.session.commit()
        return ok(visit_schema.dump(pregled), "Slike uspešno dodate i analizirane kroz UNet.")

    except Exception as e:
        db.session.rollback()
        return error(f"Greška tokom obrade medija: {str(e)}", 500)


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




@bp.get("/patient/<int:patient_id>")
def get_visits_by_patient(patient_id):
    
    pacijent = Pacijent.query.get(patient_id)
    if not pacijent:
        return not_found(f"Pacijent sa ID-jem {patient_id} ne postoji.")

    
    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 10, type=int), 100)

    
    pagination = Pregled.query.filter_by(patient_id=patient_id)\
                              .order_by(Pregled.visit_number.desc())\
                              .paginate(page=page, per_page=per_page, error_out=False)

    
    serialized_visits = visits_schema.dump(pagination.items)

    return ok({
        "patient": {
            "patient_id": pacijent.patient_id,
            "first_name": pacijent.first_name,
            "last_name": pacijent.last_name
        },
        "visits": serialized_visits,
        "total": pagination.total,
        "pages": pagination.pages,
        "current_page": page
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

    img_bytes = img_file.read()
    ml_res = ml_service.predict_glaucoma_segmentation(img_bytes)
    
    unikatno_ime = generisi_jedinstveno_ime(img_file.filename)

    img_path = os.path.join(current_app.config['IMAGES_FOLDER'], unikatno_ime)
    img_file.seek(0)
    img_file.save(img_path)

    


    mask_filename = get_mask_filename(unikatno_ime)
    mask_path = os.path.join(current_app.config['MASKS_FOLDER'], mask_filename)

    
    
    if "mask_bytes" in ml_res:
        with open(mask_path, "wb") as f:
            f.write(ml_res["mask_bytes"])
    
    return unikatno_ime, ml_res["vcdr"], ml_res.get("status", "")



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