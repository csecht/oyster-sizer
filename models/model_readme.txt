Feb 5, 2025
Using the Ultralytics package, custom training run parameters and results:

model = YOLO(yolo11n.pt)

results = model.train(
    data='oyster.yaml',
    epochs=110,
    imgsz=960,
    device=0,
    batch=20,
    name='oyster3_yolo11n_960_110e_20b',
    optimizer='auto'  # AdamW
    line_width=1,
    close_mosaic=30,
    fliplr=0,
)

110 epochs completed in 0.478 hours.
Ultralytics 8.3.38 🚀 Python-3.12.3 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 4070, 12002MiB)
YOLO11n summary (fused): 238 layers, 2,582,542 parameters, 0 gradients, 6.3 GFLOPs
                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95):
                    all        217        989      0.966      0.968      0.987       0.92
                   disk        210        337      0.991      0.983      0.995      0.991
                 oyster          8        652      0.942      0.954       0.98      0.849

oyster.yaml file contents:
    path: /home/datasets/oyster3
    train: images/train  # 1574 images
    val: images/val  # val 217 images
    names:
        0: disk
        1: oyster
