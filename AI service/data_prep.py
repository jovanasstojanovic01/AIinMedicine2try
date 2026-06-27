import pandas as pd
import numpy as np
import os
import config
from datetime import datetime, timedelta
import random

def get_val(row, col_name, is_float=True):
    if row.empty or col_name not in row.columns:
        return None
    val = row[col_name].iloc[0]
    if pd.isna(val) or val == '/' or str(val).strip() == '':
        return None
    return float(val) if is_float else val

def main():
    print("Loading core files...")
    baseline_path = os.path.join(config.OUTPUT_DIR, "grape_baseline_merged.xlsx")
    followup_path = os.path.join(config.OUTPUT_DIR, "grape_followup_merged.xlsx")
    features_path = os.path.join(config.OUTPUT_DIR, "grape_extracted_features.xlsx")

    baseline_df = pd.read_excel(baseline_path)
    followup_df = pd.read_excel(followup_path)
    features_df = pd.read_excel(features_path)

    # Pravilno mapiranje kolona iz drugog reda (kako si navela)
    baseline_mapping = {
        'Unnamed: 8': 'PLR2',
        'Unnamed: 9': 'PLR3',
        'Unnamed: 10': 'MD',
        'OCT RNFL thickness': 'Mean',
        'Unnamed: 12': 'S',
        'Unnamed: 13': 'N',
        'Unnamed: 14': 'I',
        'Unnamed: 15': 'T'
    }
    baseline_df.rename(columns=baseline_mapping, inplace=True)

    if 'Unnamed: 10' in followup_df.columns and 'MD' not in followup_df.columns:
        followup_df.rename(columns={'Unnamed: 10': 'MD'}, inplace=True)

    print("Loading names datasets...")
    female_names_path = os.path.join(config.DATA_PREP, "prep", "female_data.xlsx")
    male_names_path = os.path.join(config.DATA_PREP, "prep", "male_data.xlsx")

    female_names_df = pd.read_excel(female_names_path)
    male_names_df = pd.read_excel(male_names_path)

    # Čišćenje ID-jeva
    baseline_df = baseline_df.dropna(subset=['Subject Number'])
    followup_df = followup_df.dropna(subset=['Subject Number'])
    
    baseline_df['Subject Number'] = baseline_df['Subject Number'].astype(int)
    followup_df['Subject Number'] = followup_df['Subject Number'].astype(int)

    baseline_df.replace('/', np.nan, inplace=True)
    followup_df.replace('/', np.nan, inplace=True)

    print(f"Loaded: Baseline ({len(baseline_df)} rows), Follow-up ({len(followup_df)} rows).")

    print("\nCreating Patients table with Baseline Progression Flags (PLR2, PLR3, MD)...")
    
    # Prvo pravimo rečnik flegova za svakog pacijenta posebno za OD i OS
    patient_flags = {}
    for subj_id, group in baseline_df.groupby('Subject Number'):
        od_group = group[group['Laterality'] == 'OD']
        os_group = group[group['Laterality'] == 'OS']
        
        patient_flags[subj_id] = {
            'od_plr2': get_val(od_group, 'PLR2'),
            'od_plr3': get_val(od_group, 'PLR3'),
            'od_md': get_val(od_group, 'MD'),
            'os_plr2': get_val(os_group, 'PLR2'),
            'os_plr3': get_val(os_group, 'PLR3'),
            'os_md': get_val(os_group, 'MD')
        }

    patients_raw = baseline_df.drop_duplicates(subset=['Subject Number']).copy()
    current_year = 2026
    
    first_name_list = []
    last_name_list = []
    birth_dates = []
    baseline_exam_dates = {} 
    
    f_idx = 0
    m_idx = 0

    for idx, row in patients_raw.iterrows():
        subj_id = int(row['Subject Number'])
        gender_val = str(row.get('Gender', '')).strip().upper()
        age_val = row.get('Age', np.nan)
        
        if gender_val in ['F', 'FEMALE']:
            i = f_idx % len(female_names_df)
            first_name_list.append(female_names_df.iloc[i]['ime'])
            last_name_list.append(female_names_df.iloc[i]['prezime'])
            f_idx += 1
        elif gender_val in ['M', 'MALE']:
            i = m_idx % len(male_names_df)
            first_name_list.append(male_names_df.iloc[i]['ime'])
            last_name_list.append(male_names_df.iloc[i]['prezime'])
            m_idx += 1
        else:
            first_name_list.append("Patient")
            last_name_list.append(f"Unk-{subj_id}")

        if pd.notna(age_val) and str(age_val).replace('.','',1).isdigit():
            birth_year = current_year - int(float(age_val))
            month = random.randint(1, 12)
            day = random.randint(1, 28) if month == 2 else (random.randint(1, 30) if month in [4, 6, 9, 11] else random.randint(1, 31))
            birth_date_str = f"{birth_year}-{month:02d}-{day:02d}"
        else:
            birth_date_str = "1965-06-15"
            
        birth_dates.append(birth_date_str)
        
        start_date = datetime(random.randint(2015, 2020), random.randint(1, 12), random.randint(1, 28))
        baseline_exam_dates[subj_id] = start_date

    # Konstruisanje nove tabele pacijenata sa uključenim flegovima
    patients_table = pd.DataFrame({
        'patient_id': patients_raw['Subject Number'],
        'first_name': first_name_list,
        'last_name': last_name_list,
        'gender': patients_raw['Gender'].fillna(''),
        'birth_date': birth_dates,
        'cct': patients_raw['CCT'].fillna(0.0),
        'glaucoma_category': patients_raw['Category of Glaucoma'].fillna('OAG'),
        # Pridruživanje flegova koji su izbačeni iz pregleda
        'od_plr2': [patient_flags[sid]['od_plr2'] for sid in patients_raw['Subject Number']],
        'od_plr3': [patient_flags[sid]['od_plr3'] for sid in patients_raw['Subject Number']],
        'od_md_flag': [patient_flags[sid]['od_md'] for sid in patients_raw['Subject Number']],
        'os_plr2': [patient_flags[sid]['os_plr2'] for sid in patients_raw['Subject Number']],
        'os_plr3': [patient_flags[sid]['os_plr3'] for sid in patients_raw['Subject Number']],
        'os_md_flag': [patient_flags[sid]['os_md'] for sid in patients_raw['Subject Number']]
    })

    print("Preparing and propagating missing timeline data...")
    baseline_df['Visit Number'] = 0
    baseline_df['Interval Years'] = 0.0

    all_data_raw = pd.concat([baseline_df, followup_df], ignore_index=True)
    all_data_raw.sort_values(by=['Subject Number', 'Laterality', 'Visit Number'], inplace=True, ignore_index=True)

    # Uklanjamo 'MD' iz kolona koje se propagiraju kroz preglede jer je on sada deo pacijenta
    potential_columns = ['Corresponding CFP', 'Mean', 'S', 'N', 'I', 'T', 'vCDR', 'hCDR', 'aCDR', 'Rim_Area_Pixels']
    columns_to_fill = [col for col in potential_columns if col in all_data_raw.columns]
    
    if columns_to_fill:
        all_data_raw[columns_to_fill] = all_data_raw.groupby(['Subject Number', 'Laterality'])[columns_to_fill].ffill()

    print("Merging left and right eye records into clinical exams...")
    exam_rows = []
    multimedia_rows = []

    grouped = all_data_raw.groupby(['Subject Number', 'Visit Number'])

    for (subj_id, visit_num), group in grouped:
        od_row = group[group['Laterality'] == 'OD']
        os_row = group[group['Laterality'] == 'OS']
        
        if od_row.empty and os_row.empty:
            continue
            
        interval = group['Interval Years'].iloc[0] if pd.notna(group['Interval Years'].iloc[0]) else 0.0

        baseline_date = baseline_exam_dates.get(int(subj_id), datetime(2018, 1, 1))
        days_since_baseline = int(float(interval) * 365.25)
        actual_exam_date = baseline_date + timedelta(days=days_since_baseline)
        exam_date_str = actual_exam_date.strftime('%Y-%m-%d')

        od_iop_val = get_val(od_row, 'IOP')
        os_iop_val = get_val(os_row, 'IOP')

        max_iop = max(filter(None, [od_iop_val, os_iop_val]), default=16.0)

        # Generisanje komentara na osnovu IOP pritiska (pošto smo izbacili MD iz exam-a)
        if max_iop > 24:
            therapy_options = ["Surgery", "Enhanced drugs"]
            comments_options = [
                f"Elevated intraocular pressure detected ({max_iop} mmHg). High risk of optic nerve damage. Surgical intervention indicated.",
                f"Uncontrolled IOP at {max_iop} mmHg. Therapy escalated to combined regimens, monitor closely."
            ]
        elif max_iop > 21:
            therapy_options = ["Laser", "Enhanced drugs"]
            comments_options = [
                f"Borderline intraocular pressure ({max_iop} mmHg). SLT (selective laser trabeculoplasty) is advised.",
                f"IOP fluctuating around {max_iop} mmHg. Escalated to enhanced topical drugs configuration."
            ]
        else:
            therapy_options = ["Ništa", "Enhanced drugs"]
            comments_options = [
                f"IOP well-controlled ({max_iop} mmHg). Continue routine monitoring.",
                f"Intraocular pressure within normal limits. Glaucomatous features show no current signs of progression."
            ]

        chosen_therapy = random.choice(therapy_options)
        chosen_comment = random.choice(comments_options)

        exam_data = {
            'patient_id': int(subj_id),
            'visit_number': int(visit_num),
            'exam_date': exam_date_str,
            
            # Right Eye (OD) - BEZ MD KOLONE
            'od_iop': od_iop_val,
            'od_oct_mean': get_val(od_row, 'Mean'),
            'od_oct_s': get_val(od_row, 'S'),
            'od_oct_n': get_val(od_row, 'N'),
            'od_oct_i': get_val(od_row, 'I'),
            'od_oct_t': get_val(od_row, 'T'),
            'od_progression_status': get_val(od_row, 'Progression Status') if visit_num == 0 else None,
            
            # Left Eye (OS) - BEZ MD KOLONE
            'os_iop': os_iop_val,
            'os_oct_mean': get_val(os_row, 'Mean'),
            'os_oct_s': get_val(os_row, 'S'),
            'os_oct_n': get_val(os_row, 'N'),
            'os_oct_i': get_val(os_row, 'I'),
            'os_oct_t': get_val(os_row, 'T'),
            'os_progression_status': get_val(os_row, 'Progression Status') if visit_num == 0 else None,
            
            'physician_comment': chosen_comment,
            'therapy': chosen_therapy
        }
        exam_rows.append(exam_data)
        
        od_img = get_val(od_row, 'Corresponding CFP', is_float=False)
        os_img = get_val(os_row, 'Corresponding CFP', is_float=False)
        
        od_feat = features_df[features_df['Corresponding CFP'] == od_img] if od_img else pd.DataFrame()
        os_feat = features_df[features_df['Corresponding CFP'] == os_img] if os_img else pd.DataFrame()
        
        def determine_feat(row, feat_df, col):
            val = get_val(row, col)
            if val is not None:
                return val
            if not feat_df.empty and col in feat_df.columns:
                return float(feat_df[col].iloc[0]) if pd.notna(feat_df[col].iloc[0]) else None
            return None

        multimedia_data = {
            'patient_id': int(subj_id),
            'visit_number': int(visit_num),
            'od_image': od_img if od_img else None,
            'os_image': os_img if os_img else None,
            
            'od_vcdr': determine_feat(od_row, od_feat, 'vCDR'),
            'od_hcdr': determine_feat(od_row, od_feat, 'hCDR'),
            'od_acdr': determine_feat(od_row, od_feat, 'aCDR'),
            'od_rim_area_pixels': determine_feat(od_row, od_feat, 'Rim_Area_Pixels'),
            
            'os_vcdr': determine_feat(os_row, os_feat, 'vCDR'),
            'os_hcdr': determine_feat(os_row, os_feat, 'hCDR'),
            'os_acdr': determine_feat(os_row, os_feat, 'aCDR'),
            'os_rim_area_pixels': determine_feat(os_row, os_feat, 'Rim_Area_Pixels'),
        }
        multimedia_rows.append(multimedia_data)

    exams_table = pd.DataFrame(exam_rows)
    multimedia_table = pd.DataFrame(multimedia_rows)

    print("Merging and formatting final tables...")
    exams_table.sort_values(by=['patient_id', 'visit_number'], inplace=True, ignore_index=True)
    multimedia_table.sort_values(by=['patient_id', 'visit_number'], inplace=True, ignore_index=True)

    exams_table.insert(0, 'exam_id', exams_table.index + 1)
    multimedia_table.insert(0, 'multimedia_id', multimedia_table.index + 1)

    multimedia_table.insert(1, 'exam_id', exams_table['exam_id'])
    multimedia_table.drop(columns=['patient_id', 'visit_number'], inplace=True)

    print("Pipeline script executed successfully!")

    out_prep_dir = os.path.join(config.OUTPUT_DIR, "prep")
    if not os.path.exists(out_prep_dir):
        os.makedirs(out_prep_dir)

    patients_path = os.path.join(out_prep_dir, "table_patients.xlsx")
    exams_path = os.path.join(out_prep_dir, "table_exams.xlsx")
    multimedia_path = os.path.join(out_prep_dir, "table_multimedia.xlsx")

    patients_table.to_excel(patients_path, index=False)
    exams_table.to_excel(exams_path, index=False)
    multimedia_table.to_excel(multimedia_path, index=False)
    print("Files successfully saved into the output 'prep' directory.")

if __name__ == "__main__":
    main()