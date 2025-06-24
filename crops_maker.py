import os
import random
from faker import Faker
from PIL import Image, ImageDraw, ImageFont

fake = Faker('ru_RU')

FONTS_DIR  = 'fonts'
CROPS_DIR  = 'crops'
OUTPUT_DIR = 'dataset'

CATEGORIES = ['printed', 'handwritten']
FIELDS = {
    'name':         'first_name',
    'second_name':  'last_name',
    'third_name':   'patronymic'
}

fonts = {
    cat: [
        os.path.join(FONTS_DIR, cat, fname)
        for fname in os.listdir(os.path.join(FONTS_DIR, cat))
        if fname.lower().endswith(('.ttf', '.otf'))
    ]
    for cat in CATEGORIES
}

crops = {
    key: os.path.join(CROPS_DIR, f"{key}.png")
    for key in FIELDS
}

# Создаём выходные папки
for cat in CATEGORIES:
    for field_key in FIELDS:
        os.makedirs(os.path.join(OUTPUT_DIR, cat, field_key), exist_ok=True)

def generate_dataset(num_samples=1000):
    for i in range(num_samples):
        # Генерируем ФИО
        first       = fake.first_name()
        last        = fake.last_name()
        patronymic  = fake.middle_name()
        texts       = {'name': first, 'second_name': last, 'third_name': patronymic}

        for cat in CATEGORIES:
            font_path = random.choice(fonts[cat])
            for field_key, text in texts.items():
                bg   = Image.open(crops[field_key]).convert('RGB')
                draw = ImageDraw.Draw(bg)

                font_size = min(bg.width, bg.height)
                font      = ImageFont.truetype(font_path, font_size)
                max_w     = bg.width * 0.9

                while font.getsize(text)[0] > max_w and font_size > 10:
                    font_size -= 2
                    font = ImageFont.truetype(font_path, font_size)

                w, h      = draw.textsize(text, font=font)
                position  = ((bg.width - w) / 2, (bg.height - h) / 2)

                draw.text(position, text, font=font, fill=(0, 0, 0))

                filename = f"{i:06d}_{cat}_{field_key}.png"
                save_dir = os.path.join(OUTPUT_DIR, cat, field_key)
                bg.save(os.path.join(save_dir, filename))

if __name__ == "__main__":
    generate_dataset(num_samples=5000)
    print("Генерация датасета завершена!")
