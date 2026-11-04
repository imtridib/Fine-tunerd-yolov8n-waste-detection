# ♻️ Waste Detection with YOLOv8n

Real-time detection and classification of waste into **5 categories** (Contamination, Glass, Metal, Paper, Plastic), built by fine-tuning a pretrained **YOLOv8n** model. The model is small and fast enough for an embedded **waste-collection robot**, and it was used as the vision module for a campus robotics project (see [Real-World Application](#-real-world-application-waste-collection-robot)).

![Sample detections](images/detection_metal_05.png)

---

## 📌 Highlights

| | |
|---|---|
| **Model** | YOLOv8n (3,006,623 parameters, 8.1 GFLOPs, 6.2 MB weights) |
| **Task** | Object detection (bounding boxes + class + confidence) |
| **Classes** | Contamination, Glass, Metal, Paper, Plastic |
| **Test mAP@50** | **0.813** |
| **Test Precision / Recall** | **0.896 / 0.741** |
| **Test mAP@50-95** | **0.609** |
| **Inference speed** | ~2.0 ms/image (Tesla P100, 320×320) |
| **Training time** | ~17 minutes (50 epochs) |

---

## 📂 Repository Structure

```
.
├── waste-detection-using-yolov8n.ipynb   # Full training + evaluation + visualization notebook
├── README.md
└── images/
    ├── training_curves.png               # Loss and validation metrics per epoch
    ├── per_class_test_metrics.png        # Per-class precision / recall / mAP@50
    └── detection_*.png                   # Bounding box predictions on test images
```

---

## 🗂️ Dataset

- **Source:** Kaggle dataset `plasticwaste` (YOLO format, `data.yaml` with 5 classes)
- **Split:** 3,328 train / 998 validation / 633 test images
- **Data quality note:** Ultralytics automatically ignored images with out-of-bounds or non-normalized label coordinates: **48 train, 16 validation and 10 test** images. Effective sizes were therefore **3,280 / 982 / 623**.
- **Classes:** `Contamination`, `Glass`, `Metal`, `Paper`, `Plastic`

---

## ⚙️ Training Setup

| Parameter | Value |
|---|---|
| Base weights | `yolov8n.pt` (COCO pretrained, head re-initialised for `nc=5`) |
| Epochs | 50 |
| Image size | 320 × 320 |
| Batch size | 32 |
| Optimizer | AdamW (auto-selected, lr = 0.001111) |
| Augmentation | Ultralytics defaults (mosaic, HSV, flip, scale, translate, random erasing, Albumentations blur/CLAHE) |
| Mosaic | Turned off for the last 10 epochs (default `close_mosaic=10`) |
| Hardware | Kaggle Tesla P100 (16 GB) |
| Framework | Ultralytics 8.3.66, PyTorch 2.5.1 |

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='/kaggle/input/plasticwaste/data.yaml',
    epochs=50, batch=32, imgsz=320, workers=8,
    project='/kaggle/working/YOLOv8-Project',
    name='waste-detection', device=0, augment=True
)
metrics = model.val(data='/kaggle/input/plasticwaste/data.yaml', split='test')
```

---

## 📊 Results

### 1. Baseline vs. final model (test set)

I first ran a **1-epoch baseline** to check that the pipeline worked, then trained for **50 epochs**.

| Run | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| Baseline (1 epoch) | 0.353 | 0.454 | 0.370 | 0.228 |
| **Final (50 epochs)** | **0.896** | **0.741** | **0.813** | **0.609** |

Training for 50 epochs more than doubled mAP@50 (0.370 → 0.813) and raised precision by about 54 points.

### 2. Validation vs. test

| Split | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| Validation | 982 | 1,756 | 0.870 | 0.713 | 0.801 | 0.595 |
| Test | 623 | 1,083 | 0.896 | 0.741 | 0.813 | 0.609 |

Test scores are slightly above validation, so there is no sign of overfitting to the validation set.

### 3. Per-class results (test set)

| Class | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| **All** | 623 | 1,083 | 0.896 | 0.741 | 0.813 | 0.609 |
| Contamination | 76 | 235 | 0.791 | 0.438 | 0.531 | 0.355 |
| Glass | 146 | 194 | 0.959 | 0.943 | **0.972** | 0.797 |
| Metal | 150 | 166 | 0.927 | 0.920 | 0.935 | 0.719 |
| Paper | 58 | 87 | 0.958 | 0.621 | 0.786 | 0.589 |
| Plastic | 204 | 401 | 0.843 | 0.781 | 0.842 | 0.586 |

![Per-class test metrics](images/per_class_test_metrics.png)

### 4. Training curves

![Training curves](images/training_curves.png)

- Losses fall steadily over all 50 epochs (box loss 1.50 → 0.88, cls loss 2.77 → 0.61).
- Validation mAP@50 rises quickly during the first ~15 epochs and then plateaus near 0.79–0.80.
- Over the last 10 epochs (mosaic augmentation switched off) validation mAP@50 edges up slightly from ~0.78 to ~0.79–0.80.

### 5. Speed

| Stage | Time per image (test set) |
|---|---|
| Preprocess | 0.1 ms |
| Inference | 2.0 ms |
| Postprocess | 1.5 ms |

Measured on a Tesla P100 GPU at 320×320. Speed on a Raspberry Pi, Jetson or similar will be different and still needs to be measured.

---

## 🖼️ Sample Predictions (Bounding Boxes)

Detections from the final model on unseen test images (confidence in brackets).

### Metal
| | | |
|---|---|---|
| ![](images/detection_metal_01.png) <br> Metal (0.96) | ![](images/detection_metal_02.png) <br> Metal (0.91) | ![](images/detection_metal_03.png) <br> Metal (0.88) |
| ![](images/detection_metal_04.png) <br> Metal (0.89) | ![](images/detection_metal_05.png) <br> Metal (0.91) | ![](images/detection_metal_06.png) <br> Metal (0.91) |

### Plastic
| | | |
|---|---|---|
| ![](images/detection_plastic_02.png) <br> Plastic (0.89) | ![](images/detection_plastic_03.png) <br> Plastic (0.86) | ![](images/detection_plastic_04.png) <br> Plastic (0.84) |
| ![](images/detection_plastic_05.png) <br> Plastic (0.93) | ![](images/detection_plastic_06.png) <br> Plastic (0.93) | ![](images/detection_plastic_07.png) <br> Plastic (0.79) |

### Glass
| | |
|---|---|
| ![](images/detection_glass_01.png) <br> Glass | ![](images/detection_glass_02.png) <br> Glass (0.95) |

### Paper
| | | |
|---|---|---|
| ![](images/detection_paper_01.png) <br> Paper | ![](images/detection_paper_02.png) <br> Paper | ![](images/detection_paper_03.png) <br> Paper |

---

## 🔍 Findings & Analysis

1. **Glass and Metal are detected very reliably.** Both reach mAP@50 above 0.93 with precision and recall both around 0.92–0.96. Their distinct shape and reflective surfaces make them easy for a small model.
2. **Paper is precise but under-detected.** Precision is 0.958 but recall is 0.621, so when the model says "paper" it is usually right, but it misses many paper items. Paper has the fewest instances in the data (87 in the test set), which probably contributes.
3. **Contamination is the hardest class.** mAP@50 is 0.531 and recall is only 0.438. "Contamination" is a visually diverse catch-all category, which is hard for a 3M-parameter model to learn.
4. **Plastic is solid but not perfect.** It has the most samples (401 test instances) and mAP@50 of 0.842, but varied shapes, transparency and colours limit precision (0.843).
5. **Precision is higher than recall overall (0.896 vs 0.741).** The model is conservative: it makes few false alarms but misses some objects. For a collection robot this means few wrong pick-ups, but some waste may be skipped on the first pass.
6. **Strict-IoU performance is lower.** mAP@50-95 is 0.609 compared with 0.813 at mAP@50, so boxes are generally in the right place but not always tight. This matters if the robot arm needs accurate grasp points.
7. **Label quality matters.** Around 2% of images were dropped because of invalid bounding box coordinates. Cleaning those labels is an easy way to recover data.

---

## 🚧 Limitations

- Only 5 coarse classes. There is no fine-grained separation (e.g. PET vs. HDPE, organic waste).
- Images are resized to 320×320, which can hurt detection of small or distant objects.
- Low recall on Contamination and Paper.
- Evaluated only on the dataset's own test split. Performance in real outdoor conditions (lighting, motion blur, camera angle on the robot) has not yet been quantified.
- The reported speed is from a desktop-class GPU, not from the robot's hardware.

---

## 🚀 Future Improvements

- [ ] Fix or remove the out-of-bounds labels and retrain
- [ ] Add more Paper and Contamination samples and use class-balancing
- [ ] Train at 416 or 640 resolution and compare the accuracy/speed trade-off
- [ ] Try YOLOv8s/YOLO11n and compare with YOLOv8n
- [ ] Collect images from the robot's own camera and fine-tune on them (domain adaptation)
- [ ] Export to ONNX / TFLite / NCNN and benchmark on the robot's onboard computer
- [ ] Add a confusion matrix and PR curves to this repo

---

## ▶️ How to Run

```bash
pip install ultralytics
```

**Inference with the trained weights**
```python
from ultralytics import YOLO

model = YOLO('best_model.pt')            # weights saved by the notebook
results = model('your_image.jpg', conf=0.25)
results[0].show()                        # draws bounding boxes
```

**Retrain:** open `waste-detection-using-yolov8n.ipynb` on Kaggle, attach the `plasticwaste` dataset, and run all cells.

> Add `best_model.pt` (6.2 MB) to the repo, or to a GitHub Release, so others can run inference without retraining.

---

## 🤖 Real-World Application: Waste Collection Robot

This model was used as the AI perception component of a **waste collection robot** built by a campus mate, where I consulted on and supported the AI/ML side (model selection, training, evaluation and deployment guidance).

- 🔗 **Robot project:** [nm-razon-portfolio.vercel.app](https://nm-razon-portfolio.vercel.app/)
- **Role of the model:** detects waste in the camera feed and classifies it so the robot can decide what to collect.
- **Next step:** integration of this detector into the robot's pipeline (camera → YOLOv8n → control logic), with results to be added here.

<!-- TODO: add robot photo/demo video, hardware used, and on-robot results once available -->

---

## 🛠️ Tech Stack

Python · Ultralytics YOLOv8 · PyTorch · OpenCV · Matplotlib · Kaggle (Tesla P100)

## 👤 Author

**Niaz Morshed Razon**
Portfolio: [nm-razon-portfolio.vercel.app](https://nm-razon-portfolio.vercel.app/)

## 📜 License & Acknowledgements

- Model: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0). Check its license terms if you use this commercially.
- Dataset: Kaggle `plasticwaste` dataset. Please credit the original dataset authors and check its license.
