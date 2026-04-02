# Urban Farming -- Automated Plant Health Monitoring (Showcase)

**Course:** Deep Learning, M.Sc. Data Science & AI  
**Institution:** FH Wedel, SS 2025  
**Team Size:** Group project (team of 6)  
**Role:** Team member -- contributed to model development, evaluation, and Streamlit dashboard

> This is a showcase repository. The full source code was developed as a university group project hosted on FH Wedel's GitLab. This README documents the architecture, approach, and results.

---

## Problem

Urban farming operations need automated, real-time monitoring of plant health to detect diseases early. Manual inspection is time-consuming, inconsistent, and doesn't scale. We built a Deep Learning system that classifies plant health from images with high accuracy.

## Approach: Hierarchical Classification Pipeline

A three-stage classification approach for optimal accuracy and efficiency:

1. **Plant Identification (PlantID):** ResNet50-based classifier distinguishes between lettuce, tomato, and strawberry
2. **Binary Health Assessment:** Plant-specific models determine healthy vs. diseased status
3. **Disease Classification:** Multi-class identification when multiple disease types exist for a plant

## Key Results

| Task | Model | Accuracy |
|------|-------|----------|
| Binary Health Classification | Best CNN/ViT | **95.16%** validation accuracy |
| 16-Class Disease Classification | Best ensemble | **86.15%** test accuracy |
| Real-time Inference | Optimized pipeline | CPU and GPU deployment ready |

## Tech Stack

- **Deep Learning:** PyTorch, timm, torchvision
- **Architectures:** Vision Transformers (ViT), ResNet-50, EfficientNet-B0, MobileNetV3, ConvNeXt
- **Explainable AI:** Grad-CAM attention visualization
- **Training:** Transfer learning, AdamW optimizer, ReduceLROnPlateau scheduling
- **UI:** Streamlit dashboard with real-time predictions and explainability features
- **Data:** 31,616 curated images across 3 plant types with systematic cleaning (~30% irrelevant images removed)
- **Evaluation:** Automated architecture sweep framework for systematic model comparison

## System Architecture

```
Image Input --> PlantID (ResNet50) --> Plant Type
                                        |
                            +-----------+-----------+
                            |           |           |
                         Lettuce     Tomato    Strawberry
                            |           |           |
                    Binary Health  Binary Health  Binary Health
                    (healthy/sick) (healthy/sick) (healthy/sick)
                            |           |           |
                    Disease Class  Disease Class  Disease Class
                    (if sick)      (if sick)      (if sick)
                            |           |           |
                            +-----------+-----------+
                                        |
                                  Grad-CAM Explanation
                                        |
                                  Streamlit Dashboard
```

## My Contributions

- Developed and benchmarked multiple CNN and Transformer architectures
- Implemented Grad-CAM explainability visualizations
- Contributed to the modular training and evaluation pipeline
- Worked on the Streamlit dashboard for real-time predictions
- Co-authored the project report

## Skills Demonstrated

- Deep Learning (CNN, ViT, transfer learning)
- Computer Vision (image classification, data augmentation)
- Explainable AI (Grad-CAM)
- Model evaluation and comparison
- Production-ready code structure

---

*This project was developed as part of the Deep Learning course at FH Wedel (SS 2025). Full code is hosted on the university's GitLab.*
