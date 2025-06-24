import os
import random
from faker import Faker
from PIL import Image, ImageDraw, ImageFont

fake = Faker('ru_RU')

FONTS_DIR  = '/content/font_classifier/fonts'
CROPS_DIR  = '/content/font_classifier/crops'
OUTPUT_DIR = '/content/font_classifier/dataset'

CATEGORIES = ['printed', 'handwritten']
FIELDS = {
    'name':         'first_name',
    'second_name':  'last_name',
    'third_name':   'patronymic'
}

# Считаем шрифты
fonts = {
    cat: [
        os.path.join(FONTS_DIR, cat, fname)
        for fname in os.listdir(os.path.join(FONTS_DIR, cat))
        if fname.lower().endswith(('.ttf', '.otf'))
    ]
    for cat in CATEGORIES
}

# Считаем кропы
crops = {
    key: os.path.join(CROPS_DIR, f"{key}.png")
    for key in FIELDS
}

# Создаём выходные папки
for cat in CATEGORIES:
    for field_key in FIELDS:
        os.makedirs(os.path.join(OUTPUT_DIR, cat, field_key), exist_ok=True)

def generate_dataset(num_samples=2000):
    for i in range(num_samples):
        first      = fake.first_name()
        last       = fake.last_name()
        patronymic = fake.middle_name()
        texts      = {'name': first, 'second_name': last, 'third_name': patronymic}

        for cat in CATEGORIES:
            font_path = random.choice(fonts[cat])
            for field_key, text in texts.items():
                bg   = Image.open(crops[field_key]).convert('RGB')
                draw = ImageDraw.Draw(bg)

                # Подбираем размер шрифта
                font_size = min(bg.width, bg.height)
                max_w     = bg.width * 0.9
                font      = ImageFont.truetype(font_path, font_size)
                bbox      = draw.textbbox((0, 0), text, font=font)
                text_w    = bbox[2] - bbox[0]

                while text_w > max_w and font_size > 10:
                    font_size -= 2
                    font   = ImageFont.truetype(font_path, font_size)
                    bbox   = draw.textbbox((0, 0), text, font=font)
                    text_w = bbox[2] - bbox[0]

                # Размеры для центрирования
                text_h = bbox[3] - bbox[1]
                pos_x  = (bg.width  - text_w) / 2
                pos_y  = (bg.height - text_h) / 2

                draw.text((pos_x, pos_y), text, font=font, fill=(0, 0, 0))

                filename = f"{i:06d}_{cat}_{field_key}.png"
                save_dir = os.path.join(OUTPUT_DIR, cat, field_key)
                bg.save(os.path.join(save_dir, filename))

if __name__ == "__main__":
    generate_dataset(num_samples=5000)
    print("Генерация датасета завершена!")
