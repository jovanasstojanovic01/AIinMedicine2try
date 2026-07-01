import os
import json
import pandas as pd
from datetime import datetime
from app import create_app
from app.extensions import db
from app.models.db_models import Pacijent, Pregled, PregledMultimedija  

def parsiraj_vf_string_u_listu(vf_string):
    """
    Uzima string oblika "21,22,20,-1..." iz Excela i pretvara ga u 
    čistu Python listu brojeva spremnu za čuvanje u bazi kao JSON.
    """
    if pd.isna(vf_string) or not str(vf_string).strip():
        return None
    try:
        
        delovi = str(vf_string).split(',')
        return [int(x.strip()) if '.' not in x else float(x.strip()) for x in delovi]
    except Exception as e:
        print(f"⚠️ Greška pri parsiranju VF stringa: {str(e)}")
        return None

def pokreni_migraciju():
    app = create_app()
    with app.app_context():
        print("⏳ Kreiranje tabela i brisanje starih podataka...")
        trenutni_dir = os.path.dirname(os.path.abspath(__file__))  
        koren_projekta = os.path.dirname(trenutni_dir)             

        seed_db_folder = os.path.join(koren_projekta, "scripts")

        
        patients_path = os.path.join(seed_db_folder, "table_patients.xlsx")
        exams_path = os.path.join(seed_db_folder, "table_exams.xlsx")
        multimedia_path = os.path.join(seed_db_folder, "table_multimedia.xlsx")

        
        Pregled.query.delete()
        PregledMultimedija.query.delete()
        Pacijent.query.delete()
        db.session.commit()

        
        print("📥 Učitavanje pacijenata...")
        df_patients = pd.read_excel(patients_path)
        df_patients.columns = df_patients.columns.str.strip()

        for _, row in df_patients.iterrows():
            pacijent = Pacijent(
                patient_id=int(row['patient_id']),
                first_name=row.get('first_name', 'Pacijent'),
                last_name=row.get('last_name', f"Broj_{row['patient_id']}"),
                gender=row.get('gender', 'M'),
                birth_date=datetime.strptime(str(row['birth_date']), "%Y-%m-%d").date() if pd.notna(row.get('birth_date')) else datetime.utcnow().date(),
                cct=float(row['cct']) if pd.notna(row.get('cct')) else 540.0,
                glaucoma_category=row.get('glaucoma_category', 'None')
            )
            db.session.add(pacijent)
        db.session.flush()

        
        print("📥 Učitavanje multimedijalnih podataka (UNet rezultati)...")
        df_multimedia = pd.read_excel(multimedia_path)
        df_multimedia.columns = df_multimedia.columns.str.strip()

        for _, row in df_multimedia.iterrows():
            img_path = row.get('image_path')
            
            multimedija = PregledMultimedija(
                multimedia_id=int(row['multimedia_id']),
                image_path=img_path if pd.notna(img_path) else None,
                vcdr=float(row['vcdr']) if pd.notna(row.get('vcdr')) else 0.0,
                hcdr=float(row['hcdr']) if pd.notna(row.get('hcdr')) else 0.0,
                acdr=float(row['acdr']) if pd.notna(row.get('acdr')) else 0.0,
                rim_area_pixels=float(row['rim_area_pixels']) if pd.notna(row.get('rim_area_pixels')) else 0.0
            )
            db.session.add(multimedija)
        db.session.flush()

        
        print("📥 Učitavanje pregleda i upisivanje VF matrica iz tabela...")
        df_exams = pd.read_excel(exams_path)
        df_exams.columns = df_exams.columns.str.strip()
        
        for _, row in df_exams.iterrows():
            p_id = int(row['patient_id'])
            xml_visit_idx = int(row['visit_number']) 

            
            naziv_xml_od = f"{p_id}_{xml_visit_idx}_OD_VF.xml"
            naziv_xml_os = f"{p_id}_{xml_visit_idx}_OS_VF.xml"

            
            od_matrix_list = parsiraj_vf_string_u_listu(row.get('od_vf'))
            os_matrix_list = parsiraj_vf_string_u_listu(row.get('os_vf'))

            
            datum_str = row.get('exam_date')
            exam_date_obj = None
            if pd.notna(datum_str):
                try:
                    if isinstance(datum_str, datetime):
                        exam_date_obj = datum_str.date()
                    else:
                        exam_date_obj = datetime.strptime(str(datum_str).split()[0], "%Y-%m-%d").date()
                except ValueError:
                    exam_date_obj = datetime.utcnow().date()

            
            od_diag = row.get('od_diagnosis')
            os_diag = row.get('os_diagnosis')

            pregled = Pregled(
                exam_id=int(row['exam_id']),
                patient_id=p_id,
                visit_number=xml_visit_idx,
                exam_date=exam_date_obj,
                
                
                od_iop=float(row['od_iop']) if pd.notna(row.get('od_iop')) else None,
                od_diagnosis=od_diag if pd.notna(od_diag) and str(od_diag).strip() in ["Healthy", "Glaucoma Suspect / Positive"] else None,
                od_vf_file=naziv_xml_od if od_matrix_list else None,
                od_vf_matrix=json.dumps(od_matrix_list) if od_matrix_list else None,
                od_multimedia_id=int(row['od_multimedia_id']) if pd.notna(row.get('od_multimedia_id')) else None,
                od_next_vf_mean_pred=float(row['od_next_vf_mean_pred']) if pd.notna(row.get('od_next_vf_mean_pred')) else None,
                
                os_iop=float(row['os_iop']) if pd.notna(row.get('os_iop')) else None,
                os_diagnosis=os_diag if pd.notna(os_diag) and str(os_diag).strip() in ["Healthy", "Glaucoma Suspect / Positive"] else None,
                os_vf_file=naziv_xml_os if os_matrix_list else None,
                os_vf_matrix=json.dumps(os_matrix_list) if os_matrix_list else None,
                os_multimedia_id=int(row['os_multimedia_id']) if pd.notna(row.get('os_multimedia_id')) else None,
                os_next_vf_mean_pred=float(row['os_next_vf_mean_pred']) if pd.notna(row.get('os_next_vf_mean_pred')) else None,
                physician_comment=row.get('physician_comment') if pd.notna(row.get('physician_comment')) else None,
                therapy=row.get('therapy') if pd.notna(row.get('therapy')) else None
            )
            db.session.add(pregled)

        db.session.commit()
        print("🚀 Baza podataka je uspešno migrirana i napunjena direktno iz novih tabela!")

if __name__ == "__main__":
    pokreni_migraciju()