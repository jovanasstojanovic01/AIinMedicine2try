import cv2
import numpy as np

def extract_clinical_parameters(pred_disc, pred_cup):

    results = {
        "vCDR": 0.0,
        "hCDR": 0.0,
        "aCDR": 0.0,
        "rim_area_pixels": 0,
        "isnt_rule_valid": False,
        "quadrants_thickness": {"Inferior": 0, "Superior": 0, "Nasal": 0, "Temporal": 0},
        "diagnosis": "Healthy"
    }

    disc_img = (pred_disc * 255).astype(np.uint8)
    cup_img = (pred_cup * 255).astype(np.uint8)

    disc_contours, _ = cv2.findContours(disc_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cup_contours, _ = cv2.findContours(cup_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(disc_contours) == 0 or len(cup_contours) == 0:
        return results  

    c_disc = max(disc_contours, key=cv2.contourArea)
    c_cup = max(cup_contours, key=cv2.contourArea)

    _, _, w_disc, h_disc = cv2.boundingRect(c_disc)
    _, _, w_cup, h_cup = cv2.boundingRect(c_cup)

    # Računanje linearnih odnosa (vCDR i hCDR)
    vCDR = h_cup / max(1, h_disc)
    hCDR = w_cup / max(1, w_disc)

    # 2. Računanje površina (Area-based metrics)
    area_disc = np.sum(pred_disc)
    area_cup = np.sum(pred_cup)
    
    aCDR = area_cup / max(1, area_disc)
    rim_area = max(0, area_disc - area_cup)

    # 3. Analiza ISNT pravila (Podela na kvadrante)
    # Pronalazimo centar mase optičkog diska
    M = cv2.moments(c_disc)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = disc_img.shape[1] // 2, disc_img.shape[0] // 2

    # Merimo debljinu prstena (rastojanje od ivice kupa do ivice diska) u 4 pravca
    # Pravci (vektori kretanja od centra): 
    # Gornji (Up): y opada, Donji (Down): y raste, Levi/Desni zavise od oka (za sada uzimamo prostornu orijentaciju)
    
    def get_rim_thickness(direction_x, direction_y):
        """Pomoćna funkcija koja broji piksele prstena u određenom smeru od centra"""
        thickness = 0
        curr_x, curr_y = cX, cY
        h, w = pred_disc.shape
        
        # Krećemo se od centra diska ka spoljašnjosti
        while 0 <= curr_x < w and 0 <= curr_y < h:
            in_disc = pred_disc[curr_y, curr_x] > 0
            in_cup = pred_cup[curr_y, curr_x] > 0
            
            # Tkivo prstena je tamo gde smo unutar diska, ali VAN kupa
            if in_disc and not in_cup:
                thickness += 1
                
            # Ako smo izašli iz diska, završili smo merenje u tom pravcu
            if not in_disc and thickness > 0:
                break
                
            curr_x += direction_x
            curr_y += direction_y
        return thickness

    # Računamo debljine (pretpostavka standardne orijentacije slike)
    thick_I = get_rim_thickness(0, 1)   # Donji (Inferior)
    thick_S = get_rim_thickness(0, -1)  # Gornji (Superior)
    thick_N = get_rim_thickness(-1, 0)  # Nosni (Nasal) - pretpostavka levo oko, prilagodljivo
    thick_T = get_rim_thickness(1, 0)   # Slepoočni (Temporal)

    # Provera ISNT pravila: I >= S >= N >= T
    isnt_valid = (thick_I >= thick_S) and (thick_S >= thick_N) and (thick_N >= thick_T)

    # 4. KONAČNA DIJAGNOZA (Na osnovu kliničkog praga vCDR i ISNT pravila)
    # Ako je vCDR preveliki (klinički prag 0.65) ILI ako je drastično narušeno ISNT pravilo
    if vCDR > 0.65 or (not isnt_valid and vCDR > 0.55):
        diagnosis = "Glaucoma Suspect / Positive"
    else:
        diagnosis = "Healthy"

    # Pakovanje svih izmerenih vrednosti
    results["vCDR"] = round(float(vCDR), 3)
    results["hCDR"] = round(float(hCDR), 3)
    results["aCDR"] = round(float(aCDR), 3)
    results["rim_area_pixels"] = int(rim_area)
    results["isnt_rule_valid"] = bool(isnt_valid)
    results["quadrants_thickness"] = {
        "Inferior": thick_I,
        "Superior": thick_S,
        "Nasal": thick_N,
        "Temporal": thick_T
    }
    results["diagnosis"] = diagnosis

    return results


def calculate_dice_score(pred, target, smooth=1e-6):
    """Pomoćna funkcija za računanje standardnog Dice skora za evaluaciju segmentacije"""
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target)
    return (2. * intersection + smooth) / (union + smooth)