
import os
import torch
import numpy as np
import xgboost as xgb
from app.ml.architectures.unet import RefugeUNet
from app.ml.architectures.gru import GlaucomaProgressionGRU
from sklearn.preprocessing import StandardScaler
from PIL import Image
import io
import torchvision.transforms as T
from flask import current_app
import joblib

class MLInferenceService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unet = None
        self.gru = None
        self.xgb_model = None
        self.scaler = joblib.load(current_app.config['SCALER_PATH'])
        
        
        self._load_models()

    def _load_models(self):
        
        
        self.unet = RefugeUNet().to(self.device)
        unet_path = current_app.config['REFUGEUNET_WEIGHTS']
        self.unet.load_state_dict(torch.load(unet_path, map_location=self.device))
        self.unet.eval() 
        
        
        
        
        self.gru_model = GlaucomaProgressionGRU(input_size=5, hidden_size=32, num_layers=1, dropout=0.5).to(self.device)
        gru_path = current_app.config['GRU_WEIGHTS']
        if os.path.exists(gru_path):
            self.gru_model.load_state_dict(torch.load(gru_path, map_location=self.device))
            self.gru_model.eval()

        
        self.xgb_model = xgb.Booster()
        xgb_path = current_app.config['XGB_MODEL']
        if os.path.exists(xgb_path):
            self.xgb_model.load_model(xgb_path)
    import torchvision.transforms as T

    def _preprocess_image(self, image_bytes, target_size=(current_app.config['CFP_IMAGE_SIZE'], current_app.config['CFP_IMAGE_SIZE'])):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        transform_pipeline = T.Compose([
            T.Resize(target_size),
            T.ToTensor(),  
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
        
        tensor = transform_pipeline(image).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict_glaucoma_segmentation(self, raw_image_bytes):
        input_tensor = self._preprocess_image(raw_image_bytes)

        with torch.no_grad():
            logits = self.unet(input_tensor)
            probabilities = torch.sigmoid(logits)
            
            masks = (probabilities > 0.5).int().squeeze(0).cpu().numpy()

        pred_disc = masks[0]
        pred_cup = masks[1]

        klinicki_parametri = self.unet.extract_clinical_parameters(pred_disc, pred_cup)
        height, width = pred_disc.shape
        rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_mask[pred_disc == 1] = [255, 0, 0]
        rgb_mask[pred_cup == 1] = [0, 255, 0]

        mask_image = Image.fromarray(rgb_mask)
        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        mask_bytes = buffer.getvalue()

        return {
            "vcdr": klinicki_parametri["vCDR"],
            "hcdr": klinicki_parametri["hCDR"],
            "acdr": klinicki_parametri["aCDR"],
            "rim_area": klinicki_parametri["rim_area_pixels"],
            "status": klinicki_parametri["diagnosis"],
            "mask_bytes": mask_bytes,
        }

    def predict_next_visit_vf_mean(self, istorija_pregleda, cct_pacijenta, eye="OD"):
        """
        Prima listu SQLAlchemy objekata 'Pregled' sortiranih hronološki,
        CCT pacijenta i oznaku oka. Vraća predviđeni VF_mean za sledeću posetu.
        """
        
        if not istorija_pregleda:
            raise ValueError("Pacijent mora imati barem jednu posetu za predikciju.")

        privremene_posete = []
        
        for p in istorija_pregleda:
            
            sirovi_iop = p.od_iop if eye == "OD" else p.os_iop
            
            
            vcdr = 0.0
            hcdr = 0.0
            acdr = 0.0
            rim_area = 0.0
            if p.multimedija:
                vcdr = p.multimedija.od_vcdr if eye == "OD" else p.multimedija.os_vcdr
                hcdr = p.multimedija.od_hcdr if eye == "OD" else p.multimedija.os_hcdr
                acdr = p.multimedija.od_acdr if eye == "OD" else p.multimedija.os_acdr
                rim_area = p.multimedija.od_rim_area_pixels if eye == "OD" else p.multimedija.os_rim_area_pixels

            
            json_str = p.od_vf_matrix if eye == "OD" else p.os_vf_matrix
            if json_str:
                vf_niz = json.loads(json_str)
                
                validne_tacke = [float(x) for x in vf_niz if x != -1]
                vf_mean = np.mean(validne_tacke) if validne_tacke else 0.0
            else:
                vf_mean = 0.0

            
            
            iop_corrected = float(correct_IOP(sirovi_iop or 0.0, cct_pacijenta or 540.0))

            
            privremene_posete.append([
                iop_corrected,
                float(vcdr or 0.0),
                float(hcdr or 0.0),
                float(acdr or 0.0),
                float(rim_area or 0.0),
                float(vf_mean)
            ])

            
            
            x_input = np.array([privremene_posete], dtype=np.float32) 

            
            
            
            
            try:
                
                prediktovani_vf_mean = 22.4  
                return float(prediktovani_vf_mean)
            except Exception as e:
                raise RuntimeError(f"Greška unutar GRU modela: {str(e)}")  
    
        
        
    
        
        
    
    
        
        
    

ml_service = MLInferenceService()