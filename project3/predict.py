import csv
from pathlib import Path
from ultralytics import YOLO

def main():
    model = YOLO('runs/detect/runs/train-3/weights/best.pt')
    
    test_dir = Path('test/images')
    image_paths = sorted([p for p in test_dir.iterdir() if p.is_file()])
    
    with Path('submission.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=['image_id', 'class_id', 'x_center', 'y_center', 'width', 'height', 'confidence'],
        )
        writer.writeheader()
        
        results = model.predict(
            source=[str(p) for p in image_paths],
            conf=0.25,
            iou=0.45,
            save=False,
            verbose=False,
            device='0'
        )
        
        for idx, result in enumerate(results):
            image_id = image_paths[idx].name
            if result.boxes is None:
                continue
            for box in result.boxes:
                x_center, y_center, width, height = box.xywhn[0].tolist()
                writer.writerow({
                    'image_id': image_id,
                    'class_id': int(box.cls[0].item()),
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'confidence': float(box.conf[0].item()),
                })
    
    print("Prediction completed! Results saved to submission.csv")

if __name__ == "__main__":
    main()
