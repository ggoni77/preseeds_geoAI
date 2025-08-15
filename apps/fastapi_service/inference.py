import cv2, numpy as np
from utils import veg_mask, split_top_bottom, detect_rows, block_stats, render_processed

def infer_image(image_bytes: bytes, blocks: int = 10):
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError('Imagen inválida')

    _, bottom = split_top_bottom(img_bgr)
    mask_bottom = veg_mask(bottom)  # <- nuevo

    lines = detect_rows(bottom, mask_bottom)
    lines_list = []
    if len(lines) > 0:
        for l in lines:
            x1,y1,x2,y2 = l[0]
            lines_list.append((int(x1),int(y1),int(x2),int(y2)))

    stats = block_stats(mask_bottom, blocks=blocks)
    pct_global = float(np.round(100.0 * (mask_bottom.sum()/255) / mask_bottom.size, 2))

    processed_rgb = render_processed(img_bgr, mask_bottom)
    _, proc_png = cv2.imencode('.png', cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR))

    return {'pct_global': pct_global,'blocks_stats': stats,'row_lines': lines_list,
            'processed_png': proc_png.tobytes(),'orig_bgr': img_bgr}
