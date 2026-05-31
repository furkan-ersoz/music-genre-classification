# Column Explanations

This document explains the columns used in `mgc_literature_review_2024_2026.csv`. The table summarizes recent Music Genre Classification (MGC) studies and is intended for academic project documentation and GitHub publication.

| Column | Explanation |
|---|---|
| `Paper_ID` | Unique numeric identifier assigned to each reviewed paper. |
| `Title` | Full title of the reviewed study. |
| `APA` | Bibliographic reference of the study in APA-style format. |
| `Abstract_TR` | Turkish summary of the paper, kept for internal academic reporting and thesis/project notes. |
| `Year` | Publication year of the study. |
| `Dataset` | Dataset(s) used in the study, including dataset variants or important dataset details when available. |
| `Task_Type` | Definition of the machine learning/computer vision task addressed by the study. |
| `Split_Type` | Train/test/validation split strategy or cross-validation protocol used by the study. |
| `Evaluation_Protocol` | Detailed explanation of how the experimental evaluation was conducted. |
| `Input_Type` | Type of model input used in the experiments, such as spectrogram image, MFCC feature map, or pretrained audio embedding. |
| `Spectrogram_Type` | Specific spectrogram or time-frequency representation used, if applicable. |
| `Feature_Type` | Main feature representation used by the model, such as handcrafted audio features, spectrogram-based visual features, or SSL embeddings. |
| `Model_Type` | High-level modeling approach, such as CNN, Transformer, hybrid CNN-Transformer, GAN-based framework, or ensemble method. |
| `Backbone_Model` | Specific architecture or backbone model used in the study. |
| `Pretraining_Type` | Whether the model was trained from scratch or used pretrained/self-supervised representations. |
| `Data_Augmentation` | Data augmentation techniques used in the study, if reported. |
| `Evaluation_Metric` | Performance metrics reported in the study. |
| `Accuracy` | Reported accuracy or overall accuracy values, including dataset-specific values when multiple datasets were used. |
| `F1_Score` | Reported F1-score values. If a global F1-score was not provided, this field notes the available reporting style. |
| `Cross_Dataset_Evaluation` | Indicates whether the study evaluated the model on more than one dataset or tested cross-dataset/generalization behavior. |
| `Explainability_Method` | Explainability or interpretability method used in the study, if any. |
| `Model_Size` | Reported model size, parameter count, or architecture size information. |
| `Inference_Time` | Reported inference time or runtime information, if available. |
| `Key_Limitation` | Main limitation identified from the study. |
| `Key_Limitation2` | Additional limitation identified from the study. |

## Notes

- `Abstract_TR` is intentionally kept in Turkish.
- Empty placeholder rows from the original pasted table were removed for a cleaner GitHub-ready dataset.
- Values such as `Not reported` indicate that the reviewed paper did not clearly provide that information.
