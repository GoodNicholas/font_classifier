import onnxruntime as ort
import cv2
import numpy as np
import albumentations as A

ONNX_PATH  = 'swin_classifier.onnx'
IMAGE_PATH = '/content/font_classifier/dataset/printed/name/000000_printed_name.png'
IMG_SIZE   = 224

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # три канала
])

def preprocess(image_path):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    data = transform(image=img)['image']
    data = np.transpose(data, (2, 0, 1))
    data = data[np.newaxis, ...]
    return data.astype(np.float32)

def main():
    session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])

    inp = preprocess(IMAGE_PATH)

    outputs = session.run(['output'], {'input': inp})
    logits = outputs[0]  # shape (1,2)

    pred_idx = int(np.argmax(logits, axis=1)[0])
    labels_map = {0: 'printed', 1: 'handwritten'}
    print(f'{IMAGE_PATH} -> {labels_map[pred_idx]}')

if __name__ == '__main__':
    main()
