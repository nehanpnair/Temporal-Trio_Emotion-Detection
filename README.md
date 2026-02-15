# Emotion Detection with BERT

This project fine-tunes a pretrained BERT model for multi-class emotion classification using PyTorch.

## Exploratory Data Analysis

The dataset used for this project is `shreyaspullehf/emotion_dataset_100k`, loaded through the Hugging Face datasets library. After loading, the dataset is split into training (80%), validation (10%), and test (10%) sets using stratified sampling to maintain class proportions across splits.

A bar chart visualization of the training set reveals the distribution across 10 emotion categories. The class distribution is nearly uniform, with each emotion containing approximately 7,940 to 7,997 samples. The minimal difference between the smallest and largest class indicates that the dataset has no significant class imbalance. Despite this balanced distribution, weighted evaluation metrics are still used to ensure robust performance assessment.

## Assumptions

- The provided emotion labels are correctly annotated and representative of true emotional intent.
- The pretrained BERT model is capable of capturing contextual and semantic information necessary for emotion classification.
- Given the near-uniform class distribution, additional class-balancing techniques were not required.
- Validation accuracy is considered a reliable metric for selecting the optimal number of training epochs.

## Model Fine-tuning

The model architecture consists of a pretrained `bert-base-uncased` encoder followed by a dropout layer (0.3) and a fully connected linear classification layer that maps to the 10 emotion classes. Fine-tuning is implemented manually using PyTorch without relying on the Hugging Face Trainer API.

The training pipeline includes a custom PyTorch Dataset class that tokenizes text inputs using the BERT tokenizer with a maximum sequence length of 128 tokens. Data loaders are configured with a batch size of 16, and the model is trained for 5 epochs using the AdamW optimizer with a learning rate of 1e-5. A linear learning rate scheduler with warmup is employed to stabilize training. Cross-entropy loss is used as the training objective, with backpropagation performed through standard PyTorch calls. During training, the model's performance is monitored on the validation set, and the best-performing checkpoint is saved based on validation accuracy.

## Training Strategy and Model Selection

The model was initially trained for 3 and 4 epochs to observe validation performance trends. 
Validation accuracy continued improving through the 4th epoch without signs of overfitting. 
A 5-epoch configuration was then tested, which yielded the highest validation accuracy (0.9689). 
Therefore, the 5-epoch model was selected as the final model for evaluation on the test set.

## Evaluation Metrics

The trained model is evaluated on the held-out test set to assess its generalization performance. The evaluation reports accuracy, precision, recall, and F1-score, all using weighted averaging to account for any potential class-level variations. These metrics are computed using scikit-learn's `precision_recall_fscore_support` and `accuracy_score functions`.

## Final Test Performance

The model achieved the following results on the test set:

- **Accuracy:** 0.9669  
- **Precision (weighted):** 0.9670  
- **Recall (weighted):** 0.9669  
- **F1-score (weighted):** 0.9669  

These results indicate strong generalization performance with minimal overfitting.

A confusion matrix is generated and visualized as a heatmap to provide detailed insight into the model's class-level performance. The matrix reveals which emotion categories are correctly classified and identifies any systematic confusion patterns between specific emotion pairs.

## Observations

- Validation accuracy increased steadily across epochs, indicating stable learning.
- The model achieved strong generalization performance with minimal performance drop between validation and test sets.
- The confusion matrix shows high correct classification rates across all emotion categories.
- Fine-tuning a pretrained BERT model is highly effective for multi-class emotion classification on short-text data.

## Inference Pipeline

A prediction function `predict_text(text: str)` is implemented to enable inference on raw text inputs. The function tokenizes the input string, sets the model to evaluation mode, performs a forward pass without gradient computation, and applies softmax to obtain class probabilities. It returns both the predicted emotion label and the associated confidence score.

The inference pipeline is tested with five custom examples including sentences expressing happiness, fear, loneliness, surprise, and affection. Each example demonstrates the model's ability to correctly identify the underlying emotion and provide a confidence estimate for its prediction.

## Repository Structure

```
Temporal-Trio_Emotion-Detection/
│
├── Temporal-Trio_Emotion-Detection.ipynb   
├── README.md                               
└── requirements.txt                        
```
                      


## Contributors

This project was implemented as part of the course Natural Language Processing with Deep Learning by the team Temporal Trio.

The team members are:
- @[Neha Nair](https://github.com/nehanpnair)
- @[Niharika Paul](https://github.com/Niharika-Paul)
- @[Niharika Saha](https://github.com/niharika-saha)
