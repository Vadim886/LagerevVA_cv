import cv2
import numpy as np
import os

VIDEO_PATH = 'output.avi'
TEMPLATE_PATH = 'lagerev.png'
OUTPUT_DIR = 'out'
MATCH_THRESHOLD = 0.7

os.makedirs(OUTPUT_DIR, exist_ok=True)

template = cv2.imread(TEMPLATE_PATH)
if template is None:
    raise FileNotFoundError(f"Шаблон не найден: {TEMPLATE_PATH}")

template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Размер шаблона
th, tw = template_gray.shape[:2]

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Не удалось открыть видео: {VIDEO_PATH}")

frame_index = 0
match_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = cv2.resize(frame_gray, (template.shape[1], template.shape[0]))
    res = cv2.matchTemplate(frame_resized, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if max_val >= MATCH_THRESHOLD:
        match_count += 1
        top_left = max_loc
        bottom_right = (top_left[0] + tw, top_left[1] + th)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.imwrite(f"{OUTPUT_DIR}/match_{frame_index:04d}.png", frame)

    frame_index += 1

cap.release()
print(f"Всего совпадающих кадров: {match_count}")
