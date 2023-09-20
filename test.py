from OCRDataset import OCRDataset
import os
from mltu.utils.text_utils import ctc_decoder, get_cer
import numpy as np

dir = os.path.join("D:\\SOICT\\Handwritten OCR", 'new_public_test') 
paths = os.listdir(dir)
paths = sorted(paths, key=lambda x: int(x.split('_')[3].split('.')[0]))
image_files = [os.path.join(dir, path) for path in paths]




with open('test1.txt', 'w', encoding='utf-8') as f:
    for i, images_batch in enumerate(test_dataloader):
        images_batch = images_batch.to(device)
        outputs_batch = model(images_batch)
        
        preds_batch = ctc_decoder(
            outputs_batch.cpu().permute(1, 0, 2).detach().numpy(), 
            ocr.char_list
        )
        
        for j in range(len(preds_batch)):
            predicted_result = f"'{preds_batch[j]}'"

            f.write(f"{image_files[i * len(preds_batch) + j]} {predicted_result}\n")