## Eng Table
| Column Name              | Description                                                                                                             |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Paper_ID                 | Represents the unique identifier of the paper, typically given as an APA-style citation.                                |
| Year                     | Indicates the publication year of the study and is important for assessing the model’s recency.                         |
| Dataset                  | Specifies the dataset used (e.g., GTZAN, FMA, MTAT) and helps interpret the context of the results.                     |
| Task_Type                | Indicates whether the classification task is single-label, multi-label, or open-set, reflecting the problem complexity. |
| Split_Type               | Describes how the dataset is split (random, artist-filtered, fail-filtered), directly affecting result reliability.     |
| Evaluation_Protocol | Describes the evaluation method (cross-validation, fixed split, number of folds), ensuring fair and consistent comparison. |
| Input_Type               | Specifies whether the model uses raw audio or processed representations (e.g., spectrograms).                           |
| Spectrogram_Type         | Indicates the type of spectrogram used (Mel, Log-Mel, STFT, CQT), which can significantly impact performance.           |
| Feature_Type             | Describes the feature types used such as MFCC, Chroma, or learned embeddings.                                           |
| Model_Type               | Specifies the model family (CNN, Transformer, CRNN, etc.) and enables architectural comparison.                         |
| Backbone_Model           | Indicates the core architecture of the model (ResNet, EfficientNet, ViT, AST, etc.).                                    |
| Pretraining_Type         | Specifies whether the model is pretrained and how (supervised or self-supervised).                                      |
| Data_Augmentation        | Describes the data augmentation techniques used (SpecAugment, pitch shift, etc.) and their effect on generalization.    |
| Evaluation_Metric        | Indicates the performance metrics used in the study (Accuracy, F1-score, ROC-AUC, etc.).                                |
| Accuracy                 | Represents the overall classification accuracy, though it is not sufficient as a standalone metric.                     |
| F1_Score                 | Provides the harmonic mean of precision and recall, especially important for imbalanced datasets.                       |
| mAP                      | Mean Average Precision used in multi-label tasks to evaluate label-wise performance.                                    |
| Cross_Dataset_Evaluation | Indicates whether the model was evaluated across different datasets to measure real-world generalization.               |
| Explainability_Method    | Specifies the method used to interpret model decisions (Grad-CAM, SHAP, etc.).                                          |
| Model_Size               | Indicates the number of model parameters and helps assess computational cost.                                           |
| Inference_Time           | Represents the prediction time of the model and is important for real-time applications.                                |
| Key_Limitation           | Summarizes the main limitations reported or observed in the study.                                                      |

## Tr Table
|Column Name|	Description|
|---|---|
|Paper_ID	|Makalenin benzersiz kimliğini temsil eder ve genellikle APA formatında atıf olarak verilir.|
|Year	|Çalışmanın yayın yılıdır ve modelin güncelliğini değerlendirmek için önemlidir.|
|Dataset|	Kullanılan veri setini belirtir (örneğin GTZAN, FMA, MTAT) ve sonuçların bağlamını anlamayı sağlar.|
|Task_Type|	Sınıflandırmanın single-label, multi-label veya open-set olup olmadığını gösterir ve problemin zorluk seviyesini belirler.|
|Split_Type|	Veri setinin nasıl bölündüğünü (random, artist-filtered, fail-filtered) açıklar ve sonuçların güvenilirliğini doğrudan etkiler.|
| Evaluation_Protocol | Kullanılan değerlendirme yöntemini (cross-validation, sabit bölme, fold sayısı) açıklar ve adil karşılaştırma sağlar.           |
|Input_Type|	Modelin ham ses mi yoksa işlenmiş temsil (örneğin spektrogram) mı kullandığını belirtir.|
|Spectrogram_Type|	Kullanılan spektrogram türünü (Mel, Log-Mel, STFT, CQT) ifade eder ve performansı ciddi şekilde etkileyebilir.|
|Feature_Type	|MFCC, Chroma veya learned embedding gibi kullanılan özellik türlerini açıklar.|
|Model_Type|	Kullanılan model ailesini (CNN, Transformer, CRNN vb.) belirtir ve mimari karşılaştırma yapmayı sağlar.|
|Backbone_Model	|Modelin temel yapısını (ResNet, EfficientNet, ViT, AST vb.) gösterir.|
|Pretraining_Type|	Modelin önceden eğitilip eğitilmediğini ve nasıl eğitildiğini (supervised, self-supervised) belirtir.|
|Data_Augmentation	|Kullanılan veri artırma tekniklerini (SpecAugment, pitch shift vb.) açıklar ve genelleme performansını etkiler.|
|Evaluation_Metric|	Çalışmada kullanılan performans ölçütünü (Accuracy, F1-score, ROC-AUC vb.) belirtir.|
|Accuracy|	Modelin doğru sınıflandırma oranını ifade eder ancak tek başına yeterli bir metrik değildir.|
|F1_Score|	Precision ve recall’un dengeli ortalamasını verir ve özellikle dengesiz veri setlerinde önemlidir.|
|mAP|	Multi-label görevlerde kullanılan ortalama doğruluk ölçüsüdür ve etiket bazlı performansı gösterir.|
|Cross_Dataset_Evaluation	|Modelin farklı veri setlerinde test edilip edilmediğini belirtir ve gerçek dünya genellemesini ölçer.|
|Explainability_Method|	Modelin kararlarını açıklamak için kullanılan yöntemi (Grad-CAM, SHAP vb.) ifade eder.|
|Model_Size|	Modelin parametre sayısını belirtir ve hesaplama maliyetini anlamayı sağlar.|
|Inference_Time	|Modelin tahmin süresini gösterir ve gerçek zamanlı kullanım için önemlidir.|
|Key_Limitation	|Çalışmanın yazarları tarafından belirtilen veya gözlemlenen temel sınırlamaları özetler.|
