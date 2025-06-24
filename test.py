import torch, timm, glob, cv2
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np

ckpt = 'checkpoints/swin_best_0.9920.pt'
root = '/content/font_classifier/dataset'

# модель
model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=2)
model.load_state_dict(torch.load(ckpt, map_location='cpu'))
model.eval()

trans = A.Compose([
    A.Resize(224, 224), A.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()
])

labels_map = {0: 'printed', 1: 'handwritten'}

for path in glob.glob(f'{root}/handwritten/**/0000*.png', recursive=True)[:10]:
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    tensor = trans(image=img)['image'].unsqueeze(0)
    with torch.no_grad():
        pred = model(tensor)
    cls = labels_map[int(torch.argmax(pred))]
    print(os.path.basename(path), '->', cls)
