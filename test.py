import cv2
import os

import pytesseract
from pytesseract import Output

from orientation import orientation_detect

def is_correct_orientation (filename, orientation):
    if '180' in filename or 'original' in filename:
        return "\x1b[32mCORRECT\x1b[0m" if orientation == 'portrait' else "\x1b[31mINCORRECT\x1b[0m"
    
    elif '90' in filename or '270' in filename:
        return "\x1b[32mCORRECT\x1b[0m" if orientation == 'albumn' else "\x1b[31mINCORRECT\x1b[0m"
 

# Укажите путь к директории
directory = 'test_data'
# Перебор всех файлов в директории
for filename in os.listdir(directory):
    # Проверка, что файл имеет расширение .png
    if filename.endswith('.png'):

        # Полный путь к файлу
        file_path = os.path.join(directory, filename)
        doc = cv2.imread(file_path)
        orientation = orientation_detect(doc)
        check = is_correct_orientation(filename, orientation)
        # angle = detect_rotation_angle(doc)

        print(f'\x1b[4m{filename}\x1b[0m is \x1b[4m{orientation}\x1b[0m: it is {check}') #, rotation_angle - {angle}