import warnings, os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11m-obb.yaml')
    model.load('yolo11m-obb.pt')
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=1024,
                epochs=300,
                batch=8,
                close_mosaic=0,
                workers=4,
                optimizer='SGD',
                device='0',
                # resume=True,
                project='runs/train',
                name='exp',
                )
