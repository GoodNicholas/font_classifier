import torch
import timm

CKPT_PATH   = '/content/checkpoints/swin_best_0.9970.pt'
ONNX_PATH   = 'swin_classifier.onnx'
IMG_SIZE    = 224
DEVICE      = 'cpu'

def load_model(ckpt_path, device='cpu'):
    model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=2)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

if __name__ == '__main__':
    model = load_model(CKPT_PATH, DEVICE)

    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input':  {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f'Model exported to {ONNX_PATH}')
