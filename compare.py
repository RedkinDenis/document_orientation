"""
Это программа сравнивает скорость решения и результат с pyTesseract
Стоит отметить, что я также замеряю время которое уходит на определение ориентации текста. Аналогичной стандартной функции у pyTesseract нет
"""

import cv2
import os
import time

from orientation import orientation_detect
from pyTesseract import tesseract_detect_orientation

# Укажите путь к директории
directory = 'test_data'

all_time_my = 0
all_time_tes = 0
periods = 0

# Перебор всех файлов в директории
for filename in os.listdir(directory):
    # Проверка, что файл имеет расширение .png
    if filename.endswith('.png'):

        # Полный путь к файлу
        file_path = os.path.join(directory, filename)
        doc = cv2.imread(file_path)

        start = time.process_time()
        orientation, my_angle = orientation_detect(doc)
        end = time.process_time()
        t_my = end - start
        all_time_my += t_my

        start = time.process_time()
        tes_angle = tesseract_detect_orientation(doc)
        end = time.process_time()
        t_tes = end - start
        all_time_tes += t_tes

        periods += 1

        print(f'\x1b[4m{filename}\x1b[0m My result: \x1b[32m{my_angle}\x1b[0m; Tesseract result: \x1b[32m{my_angle}\x1b[0m')
        print(f'My time: \x1b[32m{t_my}\x1b[0m; Tesseract time: \x1b[32m{t_tes}\x1b[0m')


print(f'My mean time: \x1b[32m{all_time_my / periods}\x1b[0m; Tesseract mean time: \x1b[32m{all_time_tes / periods}\x1b[0m')


        