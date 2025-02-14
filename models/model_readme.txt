Feb 14, 2025
The Ultralytics package was used to train and tun a custom oyster model
derived from YOLOv11n.
The original best model, oyster_yolo11n_960_103e_20b, was tuned for 300
iterations with 50 epochs. The hyperparameters from the best iteration,
#233, were used to retrain that model for 300 epochs. The best results
occurred at epoch 104, and those best.pt weights are used here as the
oyster_yolo11_tuned model.

The training parameters and results are as follows:

hyperparam_args233 = {
    'lr0': 0.00242,  ignored with optimizer 'auto' = AdamW
    'lrf': 0.01009,
    'momentum': 0.92746, ignored with optimizer 'auto' = AdamW
    'weight_decay': 0.00038,
    'warmup_epochs': 1.94011,
    'warmup_momentum': 0.47865,
    'box': 5.89996,
    'cls': 0.27165,
    'dfl': 1.23119,
    'hsv_h': 0.01303,
    'hsv_s': 0.32101,
    'hsv_v': 0.40969,
    'degrees': 0.0,
    'translate': 0.15574,
    'scale': 0.75304,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.0, 0.58715,
    'bgr': 0.0,
    'mosaic': 1.0,
    'mixup': 0.0,
    'copy_paste': 0.0,
}

model = YOLO('oyster_yolo11n_960_103e_20b_best.pt')
results = model.train(data=data_yaml,
                      epochs=300,
                      imgsz=960,
                      device='0',
                      batch=20,
                      name='oyster_yolo11_tuned',
                      optimizer='AdamW',
                      close_mosaic=20,
                      patience=20,
                      **hyperparam_args233,
                      )

EarlyStopping: Training stopped early as no improvement observed in last 20 epochs. Best results observed at epoch 104, best model saved as best.pt.
To update EarlyStopping(patience=20) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.
124 epochs completed in 0.553 hours.
Ultralytics 8.3.38 🚀 Python-3.12.3 torch-2.5.1+cu124 CUDA:0 (NVIDIA GeForce RTX 4070, 12002MiB)
YOLO11n summary (fused): 238 layers, 2,582,542 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):
                   all        208       1047      0.976      0.985       0.99      0.933
                  disk        208        335      0.994      0.989      0.995      0.991
                oyster          6        712      0.959      0.982      0.984      0.876
Speed: 0.3ms preprocess, 2.0ms inference, 0.0ms loss, 2.4ms postprocess per image

From results.csv:
epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2
104,1676.76,0.2349,0.11305,0.66583,0.97644,0.985,0.98959,0.93349,0.20621,0.10559,0.65303,0.000307252,0.000307252,0.000307252
Fitness = 0.9391 <-(0.93349×0.9)+(0.98959×0.1)

oyster.yaml:
    path: /home/datasets/oyster
    train: images/train  1574 images
    val: images/val  val 217 images
    names:
        0: disk
        1: oyster
