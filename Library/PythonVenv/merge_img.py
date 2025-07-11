from PIL import Image
import os
import argparse
# import logging

def write_log(message):
    with open('example.txt', 'a', encoding='utf-8') as f:
        f.write(f"{message}\n")
write_log("Начало работы скрипта для объединения изображений в MusicNote.jpg")

# Создаём "ухо" для слушания параметров
parser = argparse.ArgumentParser()
write_log("Создан парсер аргументов")

# добавляем аргументы для ожидания
parser.add_argument("--args", type=str)
write_log("Добавлен аргумент --args для получения списка файлов")

args = parser.parse_args()
write_log("Получены аргументы командной строки")
paths = []
image_files = []

image_folder = "Images" # Путь к папке с изображениями

if args.args and args.args.strip():
    import shlex
    image_files = sorted(shlex.split(args.args))
    write_log("Полученные файлы из аргументов командной строки")
else:
    # Получаем и сортируем файлы
    image_files = sorted([f for f in os.listdir(image_folder) 
                        if f.startswith("MusicNote_") and f.endswith(".jpg")])
    # logger.debug(f"Полученные файлы(из ): {image_files}")
    write_log("Полученные файлы(___---___)")


if not image_files:
    write_log("Нет файлов для объединения!")
else:
    images = [Image.open(os.path.join(image_folder, f)) for f in image_files]

    # Определяем максимальную ширину
    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)

    # Создаем пустое изображение-результат
    result = Image.new('RGB', (max_width, total_height), (255, 255, 255))

    y_offset = 0
    for img in images:
        # Если ширина не совпадает — изменяем размер (с сохранением пропорций или растягиваем)
        if img.width != max_width:
            # Вариант 1: Растянуть по ширине (может исказить изображение)
            # resized_img = img.resize((max_width, img.height), Image.Resampling.LANCZOS)
            # Вариант 2: Добавить белые поля (раскомментировать, если нужно)
            new_img = Image.new('RGB', (max_width, img.height), (255, 255, 255))
            new_img.paste(img, ((max_width - img.width) // 2, 0))
            resized_img = new_img
        else:
            resized_img = img

        # Вставляем изображение в результат
        result.paste(resized_img, (0, y_offset))
        y_offset += resized_img.height

    # Сохраняем
    result.save("MusicNote.jpg")
    write_log("Изображения объединены в MusicNote.jpg!")