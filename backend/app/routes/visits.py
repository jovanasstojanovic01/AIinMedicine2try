
import json
import os
from flask import Blueprint, request,current_app
from marshmallow import ValidationError
from app.extensions import db
from app.models.db_models import Pacijent, Pregled, PregledMultimedija  
from app.models.schemas import visit_schema, visits_schema          
from app.ml.ml_service import ml_service
from app.utils.media_helpers import generisi_jedinstveno_ime, get_mask_filename, read_and_save_vf_xml
from app.utils.responses import ok, created, error, not_found
from datetime import datetime

bp = Blueprint("visits", __name__, url_prefix="/api/visits")



@bp.post("/<int:exam_id>/predict-progression")
def evaluate_visit_progression(exam_id):
    """
    Pokreće novi GRU model koji na osnovu istorije poseta (zaključno sa ovom)
    predviđa koliki će biti VF_mean na sledećoj poseti.
    """
    eye = request.args.get("eye", "").upper()
    if eye not in ["OD", "OS"]:
        return error("Parametar 'eye' mora biti 'OD' ili 'OS'.", 400)

    
    trenutni_pregled = Pregled.query.get(exam_id)
    if not trenutni_pregled:
        return not_found("Pregled nije pronađen.")

    pacijent = trenutni_pregled.pacijent
    
    cct_pacijenta = pacijent.cct if pacijent else 540.0

    
    pregledi_istorija = Pregled.query.filter(
        Pregled.patient_id == trenutni_pregled.patient_id,
        Pregled.visit_number <= trenutni_pregled.visit_number
    ).order_by(Pregled.visit_number).all()

    try:
        
        prediktovani_vf = ml_service.predict_next_visit_vf_mean(
            istorija_pregleda=pregledi_istorija,
            cct_pacijenta=cct_pacijenta,
            eye=eye
        )

        
        if eye == "OD":
            trenutni_pregled.od_next_vf_mean_pred = prediktovani_vf
        else:
            trenutni_pregled.os_next_vf_mean_pred = prediktovani_vf
            
        db.session.commit()

        
        return ok({
            "exam_id": exam_id,
            "eye": eye,
            "current_visit_number": trenutni_pregled.visit_number,
            "predicted_next_visit_vf_mean": round(prediktovani_vf, 2)
        }, f"Uspešno predviđen VF_mean za sledeću posetu oka {eye}.")

    except Exception as e:
        db.session.rollback()
        return error(f"Greška tokom izvršavanja GRU predikcije: {str(e)}", 500)


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
    

@bp.post("/<int:exam_id>/upload-perimetry")
def upload_visit_perimetry(exam_id):
    
    pregled = Pregled.query.get(exam_id)
    if not pregled:
        return not_found(f"Pregled sa ID-jem {exam_id} nije pronađen.")

    
    file_od = request.files.get("file_OD")
    file_os = request.files.get("file_OS")

    
    if not file_od and not file_os:
        return error("Morate poslati barem jedan XML fajl ('file_OD' ili 'file_OS').", 400)

    try:
        azurirano_desno = False
        azurirano_levo = False

        
        if file_od and file_od.filename != '':
            
            unikatno_ime_od, vf_niz_od = read_and_save_vf_xml(file_od)
            
            
            pregled.od_vf_file = unikatno_ime_od
            pregled.od_vf_matrix = json.dumps(vf_niz_od)  
            azurirano_desno = True

        
        if file_os and file_os.filename != '':
            
            unikatno_ime_os, vf_niz_os = read_and_save_vf_xml(file_os)
            
            
            pregled.os_vf_file = unikatno_ime_os
            pregled.os_vf_matrix = json.dumps(vf_niz_os)
            azurirano_levo = True

        
        db.session.commit()

        
        poruka = "Uspešno sačuvani podaci perimetrije za: "
        oci = []
        if azurirano_desno: oci.append("desno oko (OD)")
        if azurirano_levo: oci.append("levo oko (OS)")
        poruka += " i ".join(oci) + "."

        
        return ok({
            "exam_id": exam_id,
            "od_vf_file": pregled.od_vf_file,
            "os_vf_file": pregled.os_vf_file
        }, poruka)

    except ValueError as ve:
        
        db.session.rollback()
        return error(str(ve), 400)
    except Exception as e:
        
        db.session.rollback()
        return error(f"Greška tokom obrade XML fajla: {str(e)}", 500)
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
        
        if img_od and img_od.filename != '':
            img_bytes = img_od.read()
            
            ml_res_od = ml_service.predict_glaucoma_segmentation(img_bytes)
            
            
            unikatno_ime_od = generisi_jedinstveno_ime(img_od.filename)
            img_path_od = os.path.join(current_app.config['IMAGES_FOLDER'], unikatno_ime_od)
            img_od.seek(0)
            img_od.save(img_path_od)

            
            mask_name_od = get_mask_filename(unikatno_ime_od)
            mask_path_od = os.path.join(current_app.config['MASKS_FOLDER'], mask_name_od)
            if "mask_bytes" in ml_res_od:
                with open(mask_path_od, "wb") as f:
                    f.write(ml_res_od["mask_bytes"])

            
            if not pregled.od_multimedija:
                m_od = PregledMultimedija()
                db.session.add(m_od)
                db.session.flush() 
                pregled.od_multimedia_id = m_od.multimedia_id
            else:
                m_od = pregled.od_multimedija

            m_od.image_path = unikatno_ime_od
            m_od.vcdr = ml_res_od["vcdr"]
            m_od.hcdr = ml_res_od["hcdr"]
            m_od.acdr = ml_res_od["acdr"]
            m_od.rim_area_pixels = float(ml_res_od["rim_area"])
            pregled.od_diagnosis = ml_res_od["status"]

        
        if img_os and img_os.filename != '':
            img_bytes = img_os.read()
            
            ml_res_os = ml_service.predict_glaucoma_segmentation(img_bytes)
            
            
            unikatno_ime_os = generisi_jedinstveno_ime(img_os.filename)
            img_path_os = os.path.join(current_app.config['IMAGES_FOLDER'], unikatno_ime_os)
            img_os.seek(0)
            img_os.save(img_path_os)

            
            mask_name_os = get_mask_filename(unikatno_ime_os)
            mask_path_os = os.path.join(current_app.config['MASKS_FOLDER'], mask_name_os)
            if "mask_bytes" in ml_res_os:
                with open(mask_path_os, "wb") as f:
                    f.write(ml_res_os["mask_bytes"])

            
            if not pregled.os_multimedija:
                m_os = PregledMultimedija()
                db.session.add(m_os)
                db.session.flush()
                pregled.os_multimedia_id = m_os.multimedia_id
            else:
                m_os = pregled.os_multimedija

            m_os.image_path = unikatno_ime_os
            m_os.mask_path = mask_name_os
            m_os.vcdr = ml_res_os["vcdr"]
            m_os.hcdr = ml_res_os["hcdr"]
            m_os.acdr = ml_res_os["acdr"]
            m_os.rim_area_pixels = float(ml_res_os["rim_area"])
            pregled.os_diagnosis = ml_res_os["status"]

        db.session.commit()
        return ok(visit_schema.dump(pregled), "Slike za OD/OS uspešno dodate i analizirane kroz UNet.")

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



# def sacuvaj_i_analiziraj_sliku(img_file):
#     if not img_file or img_file.filename == '':
#         return None, None, 0.0

#     img_bytes = img_file.read()
#     ml_res = ml_service.predict_glaucoma_segmentation(img_bytes)
    
#     unikatno_ime = generisi_jedinstveno_ime(img_file.filename)

#     img_path = os.path.join(current_app.config['IMAGES_FOLDER'], unikatno_ime)
#     img_file.seek(0)
#     img_file.save(img_path)

    


#     mask_filename = get_mask_filename(unikatno_ime)
#     mask_path = os.path.join(current_app.config['MASKS_FOLDER'], mask_filename)

    
    
    # if "mask_bytes" in ml_res:
    #     with open(mask_path, "wb") as f:
    #         f.write(ml_res["mask_bytes"])
    
    # return unikatno_ime, ml_res["vcdr"], ml_res.get("status", "")



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