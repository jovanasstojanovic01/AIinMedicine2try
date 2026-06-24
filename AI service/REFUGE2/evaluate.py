import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# Uvoz naših lokalnih modula
from .. import config
from src import RefugeDataset, val_test_transforms, RefugeUNet, extract_clinical_parameters, calculate_dice_score

def main():
    print(f"Pokretanje evaluacije na uređaju: {config.DEVICE}")
    
    # 1. Inicijalizacija test seta podataka
    test_dataset = RefugeDataset(root_dir=config.DATA_DIR, split='test', transforms=val_test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # 2. Učitavanje istreniranog modela
    model = RefugeUNet(in_channels=3, out_channels=2).to(config.DEVICE)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"Greška: Ne postoji istrenirani model na putanji {checkpoint_path}. Prvo pokreni train.py!")
        return
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    model.eval()
    
    # Liste za čuvanje metričkih rezultata i kliničkih parametara
    all_results = []
    dice_disc_list = []
    dice_cup_list = []
    
    # 3. Evaluaciona petlja
    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(test_loader, desc="[Evaluating Test Set]")):
            images = images.to(config.DEVICE)
            
            # Predikcija modela
            outputs = model(images)
            preds = torch.sigmoid(outputs) # Prebacivanje u verovatnoće (0 do 1)
            
            # Sklanjamo batch dimenziju [1, C, H, W] -> [C, H, W] i prebacujemo u NumPy
            pred_mask = preds.squeeze(0).cpu().numpy()
            true_mask = masks.squeeze(0).numpy()
            
            # Binarizacija sa pragom 0.5 (Sve preko 50% verovatnoće postaje maska)
            pred_disc = (pred_mask[0] > 0.5).astype(np.float32)
            pred_cup = (pred_mask[1] > 0.5).astype(np.float32)
            
            true_disc = true_mask[0]
            true_cup = true_mask[1]
            
            # Računanje Dice skora (koliko su precizno pogođeni oblici u odnosu na ground truth)
            dice_disc = calculate_dice_score(pred_disc, true_disc)
            dice_cup = calculate_dice_score(pred_cup, true_cup)
            
            dice_disc_list.append(dice_disc)
            dice_cup_list.append(dice_cup)
            
            # Pozivanje naše matematičke skripte za izvlačenje kliničkih parametara
            clinical_params = extract_clinical_parameters(pred_disc, pred_cup)
            
            # Ime fajla (uzimamo iz dataset objekta preko indeksa)
            img_name = test_dataset.image_names[idx]
            
            # Pakovanje podataka za tabelu
            record = {
                "Image_Name": img_name,
                "Dice_Optic_Disc": round(dice_disc, 4),
                "Dice_Optic_Cup": round(dice_cup, 4),
                "vCDR": clinical_params["vCDR"],
                "hCDR": clinical_params["hCDR"],
                "aCDR": clinical_params["aCDR"],
                "Rim_Area_Pixels": clinical_params["rim_area_pixels"],
                "ISNT_Rule_Valid": clinical_params["isnt_rule_valid"],
                "Thickness_Inferior": clinical_params["quadrants_thickness"]["Inferior"],
                "Thickness_Superior": clinical_params["quadrants_thickness"]["Superior"],
                "Thickness_Nasal": clinical_params["quadrants_thickness"]["Nasal"],
                "Thickness_Temporal": clinical_params["quadrants_thickness"]["Temporal"],
                "Predicted_Diagnosis": clinical_params["diagnosis"]
            }
            all_results.append(record)
            
    # 4. Kreiranje Pandas DataFrame-a i čuvanje rezultata
    df = pd.DataFrame(all_results)
    report_path = os.path.join(config.OUTPUT_DIR, "glaucoma_clinical_report.csv")
    df.to_csv(report_path, index=False)
    
    # 5. Ispis finalnog uspeha na celom test setu
    print("\n================ EVALUACIJA ZAVRŠENA ================")
    print(f"Prosečan Dice Skor za Optički Disk: {np.mean(dice_disc_list):.4f}")
    print(f"Prosečan Dice Skor za Optički Kup:  {np.mean(dice_cup_list):.4f}")
    print(f"Ukupno detektovanih suspektnih/glaukom slučajeva: {sum(df['Predicted_Diagnosis'] != 'Healthy')} od {len(df)}")
    print(f"Kompletan klinički izveštaj sačuvan na: {report_path}")
    print("=====================================================")

if __name__ == "__main__":
    main()