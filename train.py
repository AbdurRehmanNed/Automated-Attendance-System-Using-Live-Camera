from ultralytics import YOLO
import torch

# Ensure that GPU is being used
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
model = YOLO("yolov8m.yaml")

# Wrap the training code in a `if __name__ == '__main__':` block
if __name__ == "__main__":
    results = model.train(data="config.yaml", epochs=20, device=device)
