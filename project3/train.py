from ultralytics import YOLO
import torch

def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data='data.yaml',
        epochs=20,
        batch=16,
        imgsz=416,
        device='0',
        workers=2,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        patience=5,
        freeze=[0],
        verbose=True,
        seed=42,
        project='runs',
        name='train',
        augment=False,
        dropout=0.0
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()
