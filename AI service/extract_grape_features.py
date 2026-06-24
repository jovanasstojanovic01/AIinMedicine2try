import os
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config
from REFUGE2 import extract_clinical_parameters, RefugeUNet

def process_grape_image(img_path, model, transform):

    if not os.path.exists(img_path):
        base, ext = os.path.splitext(img_path)
        alt_exts = [".JPG", ".jpg", ".JPEG", ".jpeg", ".png", ".PNG"] if ext.lower() in [".jpg", ".jpeg"] else [".PNG", ".png", ".bmp", ".BMP"]
        
        found = False
        for alt in alt_exts:
            alt_path = base + alt
            if os.path.exists(alt_path):
                img_path = alt_path
                found = True
                break
        
        if not found:
            print(f"[UPOZORENJE] Slika ne postoji na disku: {img_path}")
            return None
    
    try:
        img_array = np.fromfile(img_path, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None or image.size == 0:
            print(f"[UPOZORENJE] OpenCV nije uspeo da dekodira sliku: {img_path}")
            return None
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"[GREŠKA] Problem pri čitanju slike {img_path}: {str(e)}")
        return None
        
    h_orig, w_orig, _ = image.shape
    
    augmented = transform(image=image)
    img_tensor = augmented['image'].unsqueeze(0).to(config.DEVICE)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        preds = torch.sigmoid(outputs).squeeze(0).cpu().numpy()
    
    pred_disc_resized = cv2.resize((preds[0] > 0.5).astype(np.float32), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    pred_cup_resized = cv2.resize((preds[1] > 0.5).astype(np.float32), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    
    clinical_params = extract_clinical_parameters(pred_disc_resized, pred_cup_resized)
    
    return {
        "vCDR": clinical_params.get("vCDR", 0.0),
        "hCDR": clinical_params.get("hCDR", 0.0),
        "aCDR": clinical_params.get("aCDR", 0.0),
        "Rim_Area_Pixels": clinical_params.get("rim_area_pixels", 0.0) or clinical_params.get("Rim_Area_Pixels", 0.0)
    }

def main():
    grape_excel_path = os.path.join(config.DATA_DIR, "VF and clinical information.xlsx") 
    grape_images_dir = os.path.join(config.DATA_DIR, "CFPs") 
    checkpoint_path = os.path.join(config.REFUGE_MODEL, "best_model.pth")
    output_csv_path = os.path.join(config.OUTPUT_DIR, "grape_extracted_features.csv")
    
    if not os.path.exists(grape_excel_path):
        print(f"[GREŠKA] Nije pronađen GRAPE Excel fajl na putanji: {grape_excel_path}")
        return
    if not os.path.exists(checkpoint_path):
        print(f"[GREŠKA] Nije pronađen sačuvani model na putanji: {checkpoint_path}")
        return

    print("-> Korak 1: Čitanje jedinstvenih slika iz GRAPE sheeta...")
    df_baseline = pd.read_excel(grape_excel_path, sheet_name=0)
    df_followup = pd.read_excel(grape_excel_path, sheet_name=1)
    
    images_baseline = df_baseline['Corresponding CFP'].dropna().unique()
    images_followup = df_followup['Corresponding CFP'].dropna().unique()
    all_unique_images = sorted(list(set(images_baseline) | set(images_followup)))
    print(all_unique_images)
    print(f"-> Ukupno detektovano {len(all_unique_images)} jedinstvenih slika za ekstrakciju.")
    
    print(f"-> Učitavanje UNet-a sa checkpoints/best_model.pth na {config.DEVICE}...")
    model = RefugeUNet(in_channels=3, out_channels=2).to(config.DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    model.eval()
    
    val_transform = A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    extracted_features_list = []
    missing_images_log = []
    
    for img_name in tqdm(all_unique_images, desc="[UNet Obrada GRAPE Slika]"):
        img_path = os.path.join(grape_images_dir, img_name)
        
        features = process_grape_image(img_path, model, val_transform)
        
        if features is not None:
            features["Corresponding CFP"] = img_name
            extracted_features_list.append(features)
        else:
            missing_images_log.append(img_name)
            
    if missing_images_log:
        print(f"[UPOZORENJE] {len(missing_images_log)} slika iz Excela nije pronađeno u folderu {grape_images_dir}.")
        
    if extracted_features_list:
        features_df = pd.DataFrame(extracted_features_list)
        
        ordered_cols = ['Corresponding CFP', 'vCDR', 'hCDR', 'aCDR', 'Rim_Area_Pixels']
        features_df = features_df[ordered_cols]
        
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        features_df.to_csv(output_csv_path, index=False)
        
        print("\n================ UKLANJANJE I EKSTREKCIJA USPEŠNA ================")
        print(f"Generisan fajl sa parametrima: {output_csv_path}")
        print(f"Ukupno sačuvanih redova (slika): {len(features_df)}")
        print("==================================================================")
    else:
        print("[GREŠKA] Nijedna slika nije uspešno procesuirana.")

if __name__ == "__main__":
    main()