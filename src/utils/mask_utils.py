import json
import numpy as np
import cv2
from shapely import wkt

def json_to_mask(json_path, image_size=(1024, 1024), binary=True):
    """
    xBD JSON etiketlerini maske görüntüsüne dönüştürür | json, mask, xbd
    
    Args:
        json_path (str): JSON dosyasının yolu.
        image_size (tuple): Görüntü boyutu (H, W).
        binary (bool): True ise sadece bina var/yok (0/1), False ise hasar seviyeleri.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    mask = np.zeros(image_size, dtype=np.uint8)
    
    # Hasar seviyeleri eşleştirmesi
    damage_map = {
        "no-damage": 1,
        "minor-damage": 2,
        "major-damage": 3,
        "destroyed": 4,
        "un-classified": 1
    }
    
    for feature in data['features']['xy']:
        poly = wkt.loads(feature['wkt'])
        coords = np.array(list(poly.exterior.coords), dtype=np.int32)
        
        if binary:
            value = 1
        else:
            subtype = feature['properties'].get('subtype', 'no-damage')
            value = damage_map.get(subtype, 1)
            
        cv2.fillPoly(mask, [coords], value)
        
    return mask
