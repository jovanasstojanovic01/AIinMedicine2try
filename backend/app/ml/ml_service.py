
import json
import os
import torch
import numpy as np
from app.ml.architectures.unet import RefugeUNet
from app.ml.architectures.gru import GlaucomaVFProgressionGRU
from sklearn.preprocessing import StandardScaler
from PIL import Image
import io
import torchvision.transforms as T
from flask import current_app
import joblib

from app.utils.data_prep import correct_IOP

class MLInferenceService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unet = None
        self.gru = None
        self.scaler = joblib.load(current_app.config['SCALER_PATH'])
        
        
        self._load_models()

    def _load_models(self):
        
        
        self.unet = RefugeUNet().to(self.device)
        unet_path = current_app.config['REFUGEUNET_WEIGHTS']
        self.unet.load_state_dict(torch.load(unet_path, map_location=self.device))
        self.unet.eval() 
        
        
        
        
        self.gru = GlaucomaVFProgressionGRU(input_size=7, hidden_size=64, num_layers=1, dropout=0.3).to(self.device)
        gru_path = current_app.config['GRU_WEIGHTS']
        if os.path.exists(gru_path):
            self.gru.load_state_dict(torch.load(gru_path, map_location=self.device))
            self.gru.eval()
        

        
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
            Gradi sekvencu za PyTorch GRU model na osnovu hronološke istorije i
            vraca predviđeni VF_mean za sledeću posetu (t+1).
            """
            if not istorija_pregleda:
                raise ValueError("Pacijent mora imati barem jednu posetu za predikciju.")

            privremene_posete = []
            istorija_sortirano = sorted(istorija_pregleda, key=lambda x: x.visit_number)
        
            prethodni_datum = None
            for p in istorija_sortirano:
                if prethodni_datum is None or p.exam_date is None:
                    interval_years = 0.0
                else:
                    razlika_dana = (p.exam_date - prethodni_datum).days
                    interval_years = float(razlika_dana / 365.25)
                prethodni_datum = p.exam_date
                sirovi_iop = p.od_iop if eye == "OD" else p.os_iop
                
                
                vcdr, hcdr, acdr, rim_area = 0.0, 0.0, 0.0, 0.0
            
                
                m_obj = p.od_multimedija if eye == "OD" else p.os_multimedija
                
                if m_obj:
                    
                    vcdr = m_obj.vcdr
                    hcdr = m_obj.hcdr
                    acdr = m_obj.acdr
                    rim_area = m_obj.rim_area_pixels

                json_str = p.od_vf_matrix if eye == "OD" else p.os_vf_matrix
                if json_str:
                    vf_niz = json.loads(json_str)
                    validne_tacke = [float(x) for x in vf_niz if x != -1]
                    vf_mean = np.mean(validne_tacke) if validne_tacke else 0.0
                else:
                    vf_mean = 0.0
                print(vf_mean)
                
                iop_corrected = float(correct_IOP(sirovi_iop or 0.0, cct_pacijenta or 540.0))

                
                privremene_posete.append([
                    iop_corrected,
                    float(vcdr or 0.0),
                    float(hcdr or 0.0),
                    float(acdr or 0.0),
                    float(rim_area or 0.0),
                    float(vf_mean),
                    float(interval_years)
                ])

            sirovi_niz = np.array(privremene_posete, dtype=np.float32)
            skalirani_niz = self.scaler.transform(sirovi_niz)
            broj_poseta = len(privremene_posete)

            x_tensor = torch.tensor([skalirani_niz], dtype=torch.float32).to(self.device)
            
            lengths_tensor = torch.tensor([broj_poseta], dtype=torch.int64).to(self.device)

            
            with torch.no_grad():
                
                preds = self.gru(x_tensor, lengths_tensor)
                
                
                
                prediktovani_vf_mean = preds[0][-1].item()

            return float(prediktovani_vf_mean)
            
    
        
        
    
    
        
        
    

ml_service = MLInferenceService()