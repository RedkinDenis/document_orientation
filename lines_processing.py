import cv2
import numpy as np
from skimage import morphology

from image_processing import *

# ////////////////////////////////////////////////////////////////
# Инструменты поиска строк

def find_most_dense_region(image, region_height=50):
    """
    Находит участок с наибольшей плотностью пикселей (одну строку).
    
    :param image: Изображение в градациях серого.
    :param region_height: Высота участка для анализа.
    :return: Координаты участка (y_start, y_end).
    """
    # Адаптивная бинаризация
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Вертикальная проекция (сумма пикселей по строкам)
    vertical_projection = np.sum(binary, axis=1)
    
    # Находим участок с наибольшей плотностью
    max_density = 0
    best_row = 0
    
    for row in range(len(vertical_projection) - region_height):
        density = np.sum(vertical_projection[row:row + region_height])
        if density > max_density:
            max_density = density
            best_row = row
    
    return best_row, best_row + region_height

def is_mostly_black_line(line, threshold=0.02):
    """
    Проверяет, является ли строка "почти черной" (менее 2% белых пикселей).
    
    :param line: Изображение строки.
    :param threshold: Порог для процента белых пикселей (по умолчанию 2%).
    :return: True, если строка "почти черная", иначе False.
    """

    _, line = cv2.threshold(line, 127, 255, cv2.THRESH_BINARY)
    # Подсчитываем количество белых пикселей
    white_pixels = np.sum(line == 255)
    
    # Общее количество пикселей в строке
    total_pixels = line.size
    
    # Вычисляем процент белых пикселей
    white_percentage = white_pixels / total_pixels
    
    # Проверяем, меньше ли процент белых пикселей порога
    return white_percentage < threshold

def find_lines(image):
    """
    Находит строки с текстом в документе

    :param image: Изображение в градациях серого.
    :return: (lines, count)
    :lines: массив "строк"
    :count: их количество
    """

    # Удаление белых краев
    image = remove_white_margins(image)

    image = cv2.bitwise_not(image)
    # Копия изображения для закрашивания
    working_image = image.copy()

    count = 0
    lines = []

    while (True):
        # Находим участок с наибольшей плотностью белых пикселей
        y_start, y_end = find_most_dense_region(working_image, 60)

        # Вытаскиваем линию        
        line = image[y_start:y_end, :]
        
        # Если найденная линия "практически черная", прекращаем поиск
        if (is_mostly_black_line(line)):
            return (lines, count)

        lines.append(line)

        # Закрашиваем найденную строку черным цветом чтобы больше не находить ее
        working_image[y_start:y_end, :] = 0  
        count += 1

# //////////////////////////////////////////////////////////////// 
# Определение ориентации линии с помощью анализа точек и запятых

def find_dots_and_commas(image):
    """
    Находит точки и запятые в строке.
    
    :param image: Изображение строки.
    :return: Список координат точек и запятых (y-координаты).

    Оставлены некоторые закомментированые куски кода, которые еще могут пригодиться для отладки
    """
    image_cp = image.copy()

    # Бинаризация изображения (текст белый, фон черный)
    _, image_cp = cv2.threshold(image_cp, 127, 255, cv2.THRESH_BINARY)
    # image = cv2.medianBlur(image, 5)

    # kernel = np.ones((3, 3), np.uint8)
    # morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Находим контуры
    contours, _ = cv2.findContours(image_cp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создаем копию изображения для отрисовки контуров
    # output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Преобразуем в цветное изображение

    # Фильтруем контуры по размеру и рисуем их
    min_area = 10  # Минимальная площадь контура (исключает шумы)
    max_area = 40  # Максимальная площадь контура (исключает слишком большие объекты)

    dots_and_commas = []

    """
    По скольку шрифты и стили текста могут быть разными, алгоритм проводит поиск знаков препинания в некоторых "разумных" пределах
    Точки и запятые ищутся до тех пор пока размер объекта не станет слишком большим, или не будет найдена хотя бы одна
    В противном случае будет возвращен пустой массив и данная строка не будет учтена. 
    Сделано это во избежание ситуации, когда за знак препинания будет ошибочно принят совсем другой символ (например съехавшая буква)
    """
    while (len(dots_and_commas) == 0 and max_area != 70):
        for contour in contours:
            # print(1)
            area = cv2.contourArea(contour)
            
            # Игнорируем слишком маленькие и слишком большие контуры
            if min_area < area < max_area:

                x, y, w, h = cv2.boundingRect(contour)
                simb = image_cp[:, x:x+w]
                # simb_cp = simb.copy()

                # суть первой проверки: если выделить вертикальную полосу в которой предположительно 
                # находится точка и закрасить черным область в которой она располагается, то полоса должна стать черной
                simb[y:y+h, :w] = 0
                if (is_mostly_black_line(simb) and h // w <= 2 and h // w >= 1): 

                    # ///////////////////////////////////////////////////////////////////////////////////////////

                    # не до конца реализованная оптимизация поиска. Не закончена так как пропала необходимость.

                    # simb_low = simb_cp[0:y + h - h//2, :]
                    # simb_up = simb_cp[y + h//2:y + h, :]
                    # is_simb = (is_mostly_black_line(simb_up) != is_mostly_black_line(simb_low))
                    
                    # print(f"is_simb - {is_simb}")
                    # cv2.imshow("simb", simb_cp)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # if (is_simb):
                    # ///////////////////////////////////////////////////////////////////////////////////////////

                    dots_and_commas.append(y + h)  # Нижняя граница контура
                    # cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)  # Зеленый цвет, толщина линии 2
        max_area += 5

    # print(f"count of dots - {len(dots_and_commas)}")
    # show_image(output_image)

    return dots_and_commas

def is_line_upside_down(line):
    """
    Определяет, перевернут ли текст в строке.
    
    :param line: Изображение строки.
    :return: True, если текст перевернут, иначе False.
    """
    # Находим точки и запятые
    dots_and_commas = find_dots_and_commas(line)
    
    if len(dots_and_commas) == 0:
        # print("no dots and commas")
        return -1  # Эту строку не будем учитывать
    
    # Средняя позиция точек и запятых
    mean_position = np.mean(dots_and_commas)
    
    # Высота строки
    height = line.shape[0]
    
    # Если средняя позиция в верхней половине строки, текст перевернут
    if mean_position < height // 2:
        return 1 # строка перевернута
    else:
        return 0 # строка в нормальной ориентации


# //////////////////////////////////////////////////////////////// 
# Определение ориентации линии с помощью анализа верхней и нижней половин строк 
# (оказался менее надежным, поэтому был заменён)

def is_line_upside_down_old(line):
    """
    Определяет, перевернута ли строка.
    
    :param line: Изображение строки.
    :return: True, если строка перевернута, иначе False.
    """
    # Адаптивная бинаризация
    binary = cv2.adaptiveThreshold(
        line, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Вертикальная проекция (сумма пикселей по строкам)
    vertical_projection = np.sum(binary, axis=1)
    
    # Разделяем проекцию на верхнюю и нижнюю части
    height = binary.shape[0]
    upper_half = vertical_projection[:height // 2]
    lower_half = vertical_projection[height // 2:]
    
    # Нормализация гистограммы
    upper_sum = np.sum(upper_half) / len(upper_half)
    lower_sum = np.sum(lower_half) / len(lower_half)
    
    # Сравниваем нормализованные суммы
    if upper_sum > lower_sum:
        return False  # Строка в нормальной ориентации
    else:
        return True  # Строка перевернута