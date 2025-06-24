# file: train_swin.py
import torch, timm, os
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from font_dataset import FontFieldDataset
from tqdm.auto import tqdm

def main():
    root_dir = '/content/font_classifier/dataset'  # измените, если нужно
    img_size = 224
    batch_sz = 64
    epochs   = 10
    device   = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_ds = FontFieldDataset(root_dir, 'train', img_size)
    val_ds   = FontFieldDataset(root_dir, 'val',   img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_sz, shuffle=False, num_workers=4)

    # Swin-Tiny; можно заменить на swin_base_patch4_window7_224
    model = timm.create_model('swin_tiny_patch4_window7_224',
                              pretrained=True, num_classes=2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ---------- train ----------
        model.train()
        train_loss, train_correct, total = 0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs} [train]'):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(logits, 1)
            train_correct += (preds == labels).sum().item()
            total += imgs.size(0)

        train_acc = train_correct / total
        scheduler.step()

        # ---------- validation ----------
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f'Epoch {epoch}/{epochs} [val]'):
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                _, preds = torch.max(logits, 1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

        val_acc = val_correct / val_total
        print(f' Epoch {epoch}: train_acc={train_acc:.4f}  val_acc={val_acc:.4f}')

        # save best
        if val_acc > best_val_acc:
            torch.save(model.state_dict(),
                       f'checkpoints/swin_best_{val_acc:.4f}.pt')
            best_val_acc = val_acc

if __name__ == '__main__':
    main()
