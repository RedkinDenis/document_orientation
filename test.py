"""
Это программа тестировщик
Прогоняет функцию orientation_detect на всех файлах из папки test_data и сверяет ответ

(Правильный ответ должен быть заложен в имени файла, поэтому следует заполнять папку с помощью программы test_generate)
"""

import cv2
import os
import time

from orientation import orientation_detect

def is_correct_orientation (filename, angle):
    if angle == 0:
        if "original" in filename:
            return "\x1b[32mCORRECT\x1b[0m"
    elif str(angle) in filename:
        return "\x1b[32mCORRECT\x1b[0m"
    else:
        return "\x1b[31mINCORRECT\x1b[0m"
 

# Укажите путь к директории
directory = 'test_data'
# Перебор всех файлов в директории
for filename in os.listdir(directory):
    # Проверка, что файл имеет расширение .png
    if filename.endswith('.png'):

        # Полный путь к файлу
        file_path = os.path.join(directory, filename)
        doc = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        start = time.process_time()
        orientation, angle = orientation_detect(doc)
        end = time.process_time()
        t = end - start

        check = is_correct_orientation(filename, angle)

        print(f'\x1b[4m{filename}\x1b[0m rotation angle is \x1b[4m{angle}\x1b[0m: it is {check}, orientation is {orientation} \x1b[31mProcessing time: {t}\x1b[0m')