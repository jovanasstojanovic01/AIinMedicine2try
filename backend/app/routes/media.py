
import os
from flask import Blueprint, current_app, send_from_directory,request
from app.utils.media_helpers import get_mask_filename
from app.utils.responses import error, not_found
from app.models.db_models import Pregled

bp = Blueprint("media", __name__, url_prefix="/api/media")

@bp.get("/image/<string:filename>")
def serve_patient_image(filename):
    folder = current_app.config['IMAGES_FOLDER']
    if not os.path.exists(os.path.join(folder, filename)):
        return not_found("Tražena slika ne postoji na serveru.")
        
    return send_from_directory(folder, filename)

@bp.get("/mask/<string:original_filename>")
def serve_patient_mask(original_filename):
    mask_filename = get_mask_filename(original_filename)

    folder = current_app.config['MASKS_FOLDER']
    if not os.path.exists(os.path.join(folder, mask_filename)):
        return not_found("Maska za traženu sliku još uvek nije generisana.")
        
    return send_from_directory(folder, mask_filename)


@bp.get("/<int:exam_id>/perimetry/download")
def download_visit_perimetry(exam_id):
    """
    Ruta koja vraća XML fajl perimetrije za konkretan pregled na osnovu parametra 'eye'.
    Primer: GET /api/visits/12/perimetry/download?eye=OD
    """
    eye = request.args.get("eye", "").upper()
    if eye not in ["OD", "OS"]:
        return error("Parametar 'eye' mora biti 'OD' ili 'OS'.", 400)

    
    pregled = Pregled.query.get(exam_id)
    if not pregled:
        return not_found(f"Pregled sa ID-jem {exam_id} nije pronađen.")

    
    naziv_fajla = pregled.od_vf_file if eye == "OD" else pregled.os_vf_file

    
    if not_found_or_empty := (not naziv_fajla or naziv_fajla.strip() == ""):
        return error(f"Za pregled {exam_id} nije učitan XML fajl za {eye} oko.", 404)

    
    folder_sa_fajlovima = current_app.config['VF_FOLDER']

    
    if not os.path.exists(os.path.join(folder_sa_fajlovima, naziv_fajla)):
        return error("Fajl je evidentiran u bazi, ali fizički ne postoji na serveru.", 404)

    try:
        
        return send_from_directory(
            directory=folder_sa_fajlovima,
            path=naziv_fajla,
            mimetype="application/xml",
            as_attachment=True,  
            download_name=f"pacijent_{pregled.patient_id}_poseta_{pregled.visit_number}_{eye}.xml"  
        )
    except Exception as e:
        return error(f"Greška prilikom slanja fajla: {str(e)}", 500)