import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config

from REFUGE2 import RefugeUNet

def pokreni_segmentaciju_za_folder(input_folder, output_folder, weights_path):

    os.makedirs(output_folder, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Pokrećem model na uređaju: {device}")
    
    model = RefugeUNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    IMG_SIZE = 512 
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    ekstenzije = ('.jpg', '.jpeg', '.png', '.bmp')
    sve_slike = [f for f in os.listdir(input_folder) if f.lower().endswith(ekstenzije)]
    
    print(f"Pronađeno {len(sve_slike)} slika za obradu.")
    
    with torch.no_grad():
        for img_name in sve_slike:
            img_path = os.path.join(input_folder, img_name)
            
            orig_image = cv2.imread(img_path)
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            height_orig, width_orig = orig_image.shape[:2] 
            
            augmented = transform(image=orig_image)
            input_tensor = augmented['image'].unsqueeze(0).to(device)
            
            logits = model(input_tensor)
            probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            
            pred_disc = (probabilities[0] > 0.5).astype(np.uint8)
            pred_cup = (probabilities[1] > 0.5).astype(np.uint8)
            
            rgb_mask = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            rgb_mask[pred_disc == 1] = [255, 0, 0] 
            rgb_mask[pred_cup == 1] = [0, 255, 0] 
            
            rgb_mask_resized = cv2.resize(rgb_mask, (width_orig, height_orig), interpolation=cv2.INTER_NEAREST)
            
            ime_bez_ekstenzije = os.path.splitext(img_name)[0]
            mask_name = f"{ime_bez_ekstenzije}_mask.png"
            mask_path = os.path.join(output_folder, mask_name)
            
            cv2.imwrite(mask_path, cv2.cvtColor(rgb_mask_resized, cv2.COLOR_RGB2BGR))
            print(f"Sačuvano: {mask_name}")

    print("--- Obrada završena! Sve maske su uspešno sačuvane. ---")

if __name__ == "__main__":
    INPUT_DIR = os.path.join(config.DATA_DIR, "CFPs")   
    OUTPUT_DIR = os.path.join(config.OUTPUT_DIR,'masks')
    WEIGHTS = os.path.join(config.REFUGE_MODEL,"best_model.pth")

    pokreni_segmentaciju_za_folder(INPUT_DIR, OUTPUT_DIR, WEIGHTS)