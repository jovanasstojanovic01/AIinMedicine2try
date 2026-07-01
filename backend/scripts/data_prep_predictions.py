import datetime
import os
import pandas as pd
import numpy as np
import json

from app import create_app


def pokreni_predikcije_za_tabele():
    app=create_app()
    with app.app_context():
        print("⏳ Pokretanje pre-prep procesa za predikcije...")
        
        from app.ml.ml_service import MLInferenceService
        trenutni_dir = os.path.dirname(os.path.abspath(__file__))
        koren_projekta = os.path.dirname(trenutni_dir)
        seed_db_folder = os.path.join(koren_projekta, "scripts")
        
        patients_path = os.path.join(seed_db_folder, "table_patients.xlsx")
        exams_path = os.path.join(seed_db_folder, "table_exams.xlsx")
        multimedia_path = os.path.join(seed_db_folder, "table_multimedia.xlsx")
        
        
        df_patients = pd.read_excel(patients_path)
        df_exams = pd.read_excel(exams_path)
        df_multimedia = pd.read_excel(multimedia_path)
        
        df_patients.columns = df_patients.columns.str.strip()
        df_exams.columns = df_exams.columns.str.strip()
        df_multimedia.columns = df_multimedia.columns.str.strip()
        
        
        df_exams = df_exams.sort_values(by=['patient_id', 'visit_number']).reset_index(drop=True)


        ml_service = MLInferenceService()


        df_exams['od_next_vf_mean_pred'] = np.nan
        df_exams['os_next_vf_mean_pred'] = np.nan
        
        
        
        
        
        class MockMultimedia:
            def __init__(self, row):
                self.vcdr = float(row['vcdr']) if pd.notna(row.get('vcdr')) else 0.0
                self.hcdr = float(row['hcdr']) if pd.notna(row.get('hcdr')) else 0.0
                self.acdr = float(row['acdr']) if pd.notna(row.get('acdr')) else 0.0
                self.rim_area_pixels = float(row['rim_area_pixels']) if pd.notna(row.get('rim_area_pixels')) else 0.0

        class MockPregled:
            def __init__(self, exam_row, mm_df):
                self.visit_number = int(exam_row['visit_number'])
                self.od_iop = exam_row.get('od_iop')
                self.os_iop = exam_row.get('os_iop')
                self.od_vf_matrix = self._pretvori_u_json_str(exam_row.get('od_vf'))
                self.os_vf_matrix = self._pretvori_u_json_str(exam_row.get('os_vf'))
                
                datum_raw = exam_row.get('exam_date')
                if pd.notna(datum_raw):
                    
                    if isinstance(datum_raw, datetime.datetime):
                        self.exam_date = datum_raw.date()
                    elif isinstance(datum_raw, datetime.date):
                        self.exam_date = datum_raw
                    else:
                        
                        self.exam_date = datetime.datetime.strptime(str(datum_raw).split()[0], "%Y-%m-%d").date()
                else:
                    self.exam_date = None
                od_mm_id = exam_row.get('od_multimedia_id')
                os_mm_id = exam_row.get('os_multimedia_id')
                
                od_mm_row = mm_df[mm_df['multimedia_id'] == od_mm_id] if pd.notna(od_mm_id) else pd.DataFrame()
                os_mm_row = mm_df[mm_df['multimedia_id'] == os_mm_id] if pd.notna(os_mm_id) else pd.DataFrame()
                
                self.od_multimedija = MockMultimedia(od_mm_row.iloc[0]) if not od_mm_row.empty else None
                self.os_multimedija = MockMultimedia(os_mm_row.iloc[0]) if not os_mm_row.empty else None
                #self.multimedija = None
            def _pretvori_u_json_str(self, vf_string):
                if pd.isna(vf_string) or not str(vf_string).strip():
                    return None
                delovi = str(vf_string).split(',')
                niz = [int(x.strip()) if '.' not in x else float(x.strip()) for x in delovi]
                return json.dumps(niz)

        
        for patient_id, group in df_exams.groupby('patient_id'):
            patient_row = df_patients[df_patients['patient_id'] == patient_id]
            if patient_row.empty:
                continue
            cct = float(patient_row.iloc[0]['cct']) if pd.notna(patient_row.iloc[0]['cct']) else 540.0
            
            istorija_pregleda_od = []
            istorija_pregleda_os = []
            
            
            for idx, exam_row in group.iterrows():
                mock_p = MockPregled(exam_row, df_multimedia)
                
                
                istorija_pregleda_od.append(mock_p)
                istorija_pregleda_os.append(mock_p)
                
                try:
                    
                    pred_od = ml_service.predict_next_visit_vf_mean(istorija_pregleda_od, cct, eye="OD")
                    df_exams.at[idx, 'od_next_vf_mean_pred'] = pred_od
                except Exception as e:
                    print(f"⚠️ Nije generisana predikcija za OD, pacijent {patient_id}, poseta {exam_row['visit_number']}: {e}")
                    
                try:
                    pred_os = ml_service.predict_next_visit_vf_mean(istorija_pregleda_os, cct, eye="OS")
                    df_exams.at[idx, 'os_next_vf_mean_pred'] = pred_os
                except Exception as e:
                    print(f"⚠️ Nije generisana predikcija za OS, pacijent {patient_id}, poseta {exam_row['visit_number']}: {e}")

        
        nove_predikcije_path = os.path.join(seed_db_folder, "table_exams_with_predictions.xlsx")
        df_exams.to_excel(nove_predikcije_path, index=False)
        
        print(f"✅ Predikcije uspešno izračunate i sačuvane u NOVOM fajlu: {nove_predikcije_path}!")
        
if __name__ == "__main__":
    pokreni_predikcije_za_tabele()