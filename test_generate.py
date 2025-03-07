import cv2
import os

# Укажите путь к папке с изображениями
input_directory = 'original_documents'
# Укажите путь к папке для сохранения результатов
output_directory = 'test_data'

# Углы поворота (в градусах)
angles = [90, 180, 270]

# Создание выходной директории, если она не существует
os.makedirs(output_directory, exist_ok=True)

# Перебор всех PNG-файлов в директории
for filename in os.listdir(input_directory):
    if filename.endswith('.png'):
        # Полный путь к файлу
        file_path = os.path.join(input_directory, filename)
        
        # Загрузка изображения
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        # Проверка, что изображение загружено успешно
        if image is None:
            print(f"Ошибка: Не удалось загрузить изображение {file_path}")
            continue
        
        # Сохранение оригинального изображения (опционально)
        original_output_path = os.path.join(output_directory, f'original_{filename}')
        cv2.imwrite(original_output_path, image)
        
        # Поворот изображения на каждый угол
        for angle in angles:
            # Получение размеров изображения
            (height, width) = image.shape[:2]
            
            # Вычисление центра изображения
            center = (width // 2, height // 2)
            
            # Получение матрицы поворота
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Вычисление новых размеров изображения после поворота
            if angle % 180 == 90:  # Для 90° и 270° меняем ширину и высоту местами
                new_width = height
                new_height = width
            else:
                new_width = width
                new_height = height
            
            # Учет изменения размеров в матрице поворота
            rotation_matrix[0, 2] += (new_width - width) / 2
            rotation_matrix[1, 2] += (new_height - height) / 2
            
            # Применение поворота с новыми размерами
            rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
            
            # Сохранение повернутого изображения
            rotated_output_path = os.path.join(output_directory, f'rotated_{angle}_{filename}')
            cv2.imwrite(rotated_output_path, rotated_image)
            
            print(f"Сохранено: {rotated_output_path}")


print("Обработка завершена!")