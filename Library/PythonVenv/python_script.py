### UnionAll поколение 2´ - 

### Решить:
### Убрать Лиги(см. DeepSeek "Что такое MIDI")(паузы)
### РЕШЕНО:
### Задача: разобраться в коде, чтобы понять, почему CNN определяет все ноты как half_note(предположение:
### потому что CNN берет изображения только головок нот(часть 1), а не изображения нот полностью(часть 2))
#  
### Код работает, но почему-то у меня ноты  сохраняются в файл detected_notes_eazy
### в порядке не слева направо, а в случайном(решение: сортировка по координате x). В DeepSeek
### все это вбил в чате "OpenCV_LastUpdate_RemoveLittleRect" все есть.

import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import mido
from mido import MidiFile, MidiTrack, Message
import shutil

# Константы
PPQ = 480  # Количество тиков на четвертную ноту
NOTE_VELOCITY = 64  # Громкость нот

# --- Часть 1: Ваш код для поиска нотных головок ---
def merge_close_rectangles(rectangles, max_distance_x=20, max_distance_y=10):
    merged = []
    for rect in rectangles:
        x, y, w, h = rect
        found = False
        for i, mrg in enumerate(merged):
            mx, my, mw, mh = mrg
            if (abs(x - mx) < max_distance_x and abs(y - my) < max_distance_y):
                new_x = min(x, mx)
                new_y = min(y, my)
                new_w = max(x + w, mx + mw) - new_x
                new_h = max(y + h, my + mh) - new_y
                merged[i] = (new_x, new_y, new_w, new_h)
                found = True
                break
        if not found:
            merged.append(rect)
    return merged

# Загрузка изображения и предобработка
img = cv2.imread('MusicNote.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Копия для поиска линий нотоносца
lines_image = gray.copy()
_, staff_bin = cv2.threshold(lines_image, 150, 255, cv2.THRESH_BINARY_INV)

# Поиск линий нотоносца
staff_lines = []
lines = cv2.HoughLinesP(staff_bin, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
if lines is not None:
    y_coords = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) < 5:  # Фильтр горизонтальных линий
            y_coords.extend([y1, y2])
    
    # Группировка близких линий
    y_coords = sorted(y_coords)
    clusters = []
    current_cluster = [y_coords[0]]
    for y in y_coords[1:]:
        if y - current_cluster[-1] < 5:
            current_cluster.append(y)
        else:
            clusters.append(current_cluster)
            current_cluster = [y]
    clusters.append(current_cluster)
    
    # Выбираем 5 самых частых линий
    clusters = sorted(clusters, key=len, reverse=True)[:5]
    staff_lines = [int(np.mean(c)) for c in clusters]
    staff_lines = sorted(staff_lines)

# Удаление линий нотоносца с основного изображения
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(binary, (x1, y1), (x2, y2), 0, 3)

# Морфологическая обработка
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

# Поиск контуров
contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Первичная фильтрация и сбор прямоугольников
rectangles = []
min_area = 20
max_area = 500

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    if min_area < area < max_area:
        rectangles.append((x, y, w, h))

# Объединение близких прямоугольников
merged_rects = merge_close_rectangles(rectangles, max_distance_x=20, max_distance_y=10)

# Финальная фильтрация по соотношению сторон
aspect_ratio_threshold = 3
result_rects = []
for rect in merged_rects:
    x, y, w, h = rect
    aspect_ratio = max(w, h) / min(w, h)
    if aspect_ratio < aspect_ratio_threshold:
        result_rects.append(rect)

# Сортируем прямоугольники по X координате (слева направо)
result_rects = sorted(result_rects, key=lambda r: r[0]) # ЗДЕСЬ

# Функция определения MIDI номера ноты
def get_note_midi(y_center, staff_lines):
    if len(staff_lines) != 5:
        return None
    
    line_dist = np.mean([staff_lines[i+1] - staff_lines[i] for i in range(4)])
    
    # Проверка линий
    for i in range(5):
        if abs(y_center - staff_lines[i]) < line_dist * 0.3:
            return [64, 67, 71, 74, 77][::-1][i]  # MIDI для линий
    
    # Проверка промежутков
    for i in range(4):
        space_center = (staff_lines[i] + staff_lines[i+1]) / 2
        if abs(y_center - space_center) < line_dist * 0.3:
            return [65, 69, 72, 76][::-1][i]  # MIDI для промежутков
    
    # Проверка за пределами
    if y_center < staff_lines[0] - line_dist * 0.5:
        return 79  # G5
    elif y_center > staff_lines[0] + line_dist * 0.5:
        return 62  # D4
    
    return None

# --- Часть 2: Сохранение изображений нот для классификации ---
# Создаем папку для сохранения нот, если ее нет
output_folder = 'detected_notes_eazy'
os.makedirs(output_folder, exist_ok=True)

# Загружаем изображение
image = cv2.imread('MusicNote.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Параметры фильтрации нот
min_note_width = 10    # Минимальная ширина ноты
min_note_height = 10   # Минимальная высота ноты

# Адаптивная бинаризация
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY_INV, 11, 2)

# Удаляем линии нотоносца
lines = cv2.HoughLinesP(binary, 1, 3.1415/180, 100, minLineLength=100, maxLineGap=10)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(binary, (x1, y1), (x2, y2), 0, 3)

# Морфологическая обработка для объединения близких элементов
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17,18))
processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Находим контуры
contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Список для хранения всех прямоугольников
rectangles = []

# Первоначальное выделение всех прямоугольников с проверкой размера
min_width = 1  # Минимальная ширина прямоугольника
min_height = 1  # Минимальная высота прямоугольника
min_area = 10   # Минимальная площадь прямоугольника

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Проверяем размеры прямоугольника
    if w >= min_width and h >= min_height and (w * h) >= min_area:
        rectangles.append((x, y, x+w, y+h))  # Сохраняем как (x1,y1,x2,y2)

# Функция для объединения пересекающихся прямоугольников
def merge_rectangles(rects):
    changed = True
    while changed:
        changed = False
        new_rects = []
        for rect in rects:
            merged = False
            for i, new_rect in enumerate(new_rects):
                # Если прямоугольники пересекаются
                if not (rect[2] < new_rect[0] or rect[0] > new_rect[2] or 
                        rect[3] < new_rect[1] or rect[1] > new_rect[3]):
                    # Объединяем в один прямоугольник
                    new_rects[i] = (min(rect[0], new_rect[0]), 
                                   min(rect[1], new_rect[1]),
                                   max(rect[2], new_rect[2]), 
                                   max(rect[3], new_rect[3]))
                    merged = True
                    changed = True
                    break
            if not merged:
                new_rects.append(rect)
        rects = new_rects
    return rects

# Объединяем прямоугольники
merged_rects = merge_rectangles(rectangles)

# Новая функция для фильтрации прямоугольников по размеру
def filter_rectangles(rects, min_width=20, min_height=20):
    filtered = []
    for rect in rects:
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        if w >= min_width and h >= min_height:
            filtered.append(rect)
    return filtered

# Фильтруем слишком маленькие области
filtered_rects = filter_rectangles(merged_rects, min_note_width, min_note_height)

# Сортируем прямоугольники по координате X (слева направо)
filtered_rects.sort(key=lambda x: x[0])

# Сохраняем каждую ноту в отдельный файл
for i, rect in enumerate(filtered_rects):
    # Получаем область с нотой (добавляем небольшой отступ)
    padding = 5  # Отступ в пикселях
    x1 = max(0, rect[0] - padding)
    y1 = max(0, rect[1] - padding)
    x2 = min(image.shape[1], rect[2] + padding)
    y2 = min(image.shape[0], rect[3] + padding)
    
    note_img = image[y1:y2, x1:x2]
    
    # Сохраняем изображение ноты
    note_filename = os.path.join(output_folder, f'note_{i+1}.png')
    cv2.imwrite(note_filename, note_img)
    
    # Также рисуем прямоугольники на исходном изображении (для визуализации)
    cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)

# Сохраняем изображение с выделенными нотами
# cv2.imwrite('notes_detected.jpg', image)

print(f"Сохранено {len(filtered_rects)} нот в папку '{output_folder}'")

# --- Часть 3: Классификация нот CNN ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*7*7, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

def predict_note_with_confidence(image_path, confidence_threshold=0.7):
    # Классы и соответствующие длительности в тиках
    classes = {
        "whole_note": PPQ*4,  # Целая нота (1920 тиков)
        "half_note_with_stem": PPQ*2,  # Половиная нота (960 тиков)
        "quarter_note": PPQ  # Четвертная нота (480 тиков)
    }
    
    # Загрузка и преобразование изображения
    img = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = transform(img).unsqueeze(0)
    
    # Предсказание
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        max_prob, predicted = torch.max(probabilities, 1)
        max_prob = max_prob.item()
    
    class_names = list(classes.keys())
    
    if max_prob < confidence_threshold:
        return "unknown", 0, max_prob
    else:
        note_type = class_names[predicted.item()]
        return note_type, classes[note_type], max_prob

# Инициализация модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("last_note_classifier.pth", map_location=device))
model.eval()

# --- Интеграция всех частей ---
# 1. Получаем MIDI номера нот
midi_notes = []
for rect in result_rects:
    x, y, w, h = rect
    y_center = y + h // 2
    midi = get_note_midi(y_center, staff_lines)
    if midi is not None:
        midi_notes.append(midi)

# 2. Получаем длительности нот

durations = []
print(result_rects)
note_files = sorted(os.listdir(output_folder), key=lambda x: int(x.split('_')[1].split('.')[0]))
print("Названия файлов:", note_files)
for note_file in note_files:
    note_path = os.path.join(output_folder, note_file)
    note_type, duration, _ = predict_note_with_confidence(note_path)
    print(note_type)
    durations.append(duration)

# 3. Создаем MIDI файл
mid = MidiFile(ticks_per_beat=PPQ)
track = MidiTrack()
mid.tracks.append(track)

# Добавляем ноты в трек
time = 1  # Начальное время
for note, duration in zip(midi_notes, durations):
    track.append(Message('note_on', note=note, velocity=NOTE_VELOCITY, time=time))
    track.append(Message('note_off', note=note, velocity=NOTE_VELOCITY, time=duration))
    time = 0  # Следующая нота начинается сразу после предыдущей

# Сохраняем MIDI файл
mid.save('AAAnote.mid')
print("MIDI файл успешно создан!")
print("MIDI номера нот:", midi_notes)
print("Длительности нот (в тиках):", durations)

## Удаляем все папки и файлы, созданные ранее(КОСТЫЛЬ, ПОФИКСИТЬ!!!!!)

# Удаление папки
dir_path = output_folder
if os.path.exists(dir_path):
    shutil.rmtree(dir_path)  # удалит папку и ВСЁ её содержимое

# Удаление файла
# file_path = "путь/к/файлу.txt"
# if os.path.exists(file_path):  # проверка, существует ли файл
#     os.remove(file_path)       # удаление файла