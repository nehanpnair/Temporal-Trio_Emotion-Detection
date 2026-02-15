# Emotion Detection with BERT

This project fine-tunes a pretrained BERT model for multi-class emotion classification using PyTorch.

## 1. Exploratory Data Analysis

The dataset used for this project is `shreyaspullehf/emotion_dataset_100k`, loaded through the Hugging Face datasets library. After loading, the dataset is split into training (80%), validation (10%), and test (10%) sets using stratified sampling to maintain class proportions across splits.

A bar chart visualization of the training set reveals the distribution across 10 emotion categories. The class distribution is nearly uniform, with each emotion containing approximately 7,940 to 7,997 samples. The minimal difference between the smallest and largest class indicates that the dataset has no significant class imbalance. Despite this balanced distribution, weighted evaluation metrics are still used to ensure robust performance assessment.

## 2. Model Fine-tuning

The model architecture consists of a pretrained `bert-base-uncased` encoder followed by a dropout layer (0.3) and a fully connected linear classification layer that maps to the 10 emotion classes. Fine-tuning is implemented manually using PyTorch without relying on the Hugging Face Trainer API.

The training pipeline includes a custom PyTorch Dataset class that tokenizes text inputs using the BERT tokenizer with a maximum sequence length of 128 tokens. Data loaders are configured with a batch size of 16, and the model is trained for 5 epochs using the AdamW optimizer with a learning rate of 1e-5. A linear learning rate scheduler with warmup is employed to stabilize training. Cross-entropy loss is used as the training objective, with backpropagation performed through standard PyTorch calls. During training, the model's performance is monitored on the validation set, and the best-performing checkpoint is saved based on validation accuracy.

## 3. Evaluation Metrics

The trained model is evaluated on the held-out test set to assess its generalization performance. The evaluation reports accuracy, precision, recall, and F1-score, all using weighted averaging to account for any potential class-level variations. These metrics are computed using scikit-learn's metric functions.

A confusion matrix is generated and visualized as a heatmap to provide detailed insight into the model's class-level performance. The matrix reveals which emotion categories are correctly classified and identifies any systematic confusion patterns between specific emotion pairs.

## 4. Inference Pipeline

A prediction function `predict_text(text: str)` is implemented to enable inference on raw text inputs. The function tokenizes the input string, sets the model to evaluation mode, performs a forward pass without gradient computation, and applies softmax to obtain class probabilities. It returns both the predicted emotion label and the associated confidence score.

The inference pipeline is tested with five custom examples including sentences expressing happiness, fear, loneliness, surprise, and affection. Each example demonstrates the model's ability to correctly identify the underlying emotion and provide a confidence estimate for its prediction.
