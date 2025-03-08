import cv2
import numpy as np

def resize_image_proportional(image, max_width=None, max_height=None):
    """
    Пропорционально уменьшает изображение до заданной ширины или высоты.

    :param image: Исходное изображение (в формате NumPy array).
    :param max_width: Максимальная ширина. Если None, ширина не ограничена.
    :param max_height: Максимальная высота. Если None, высота не ограничена.
    :return: Уменьшенное изображение.

    P.s. Данная функция использовалась для отладки, чтобы уместить изображение на мониторе
    """
    # Получаем текущие размеры изображения
    height, width = image.shape[:2]

    # Если ни ширина, ни высота не заданы, возвращаем исходное изображение
    if max_width is None and max_height is None:
        return image

    # Рассчитываем коэффициент масштабирования
    scale = 1.0
    if max_width is not None and width > max_width:
        scale = max_width / width
    if max_height is not None and height > max_height:
        scale = min(scale, max_height / height)

    # Если изображение уже меньше заданных размеров, возвращаем его
    if scale >= 1.0:
        return image

    # Вычисляем новые размеры
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Масштабируем изображение
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_image

def remove_white_margins(image):
    """
    Удаляет пустые области вокруг текста.
    """
    # Бинаризация изображения
    _, binary = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY_INV)
    
    # Нахождение контуров
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Нахождение ограничивающей рамки для всех контуров
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    
    # Обрезка изображения
    cropped = image[y:y+h, x:x+w]
    
    return cropped


def rotate_image_90(image):
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return rotated_image

def show_image(image, name="image"):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()