# Large Language Model Project
> Tashrif Mahmud
🤗: [Hugging Face Repo](https://huggingface.co/tashrifmahmud/sentiment_analysis_model_v2) | :bar_chart: [Streamlit App](https://sentiment-analysis-v2.streamlit.app/) | :film_strip: [YouTube Video](https://www.youtube.com/watch?v=a1PKN0u6dso)

## Project Overview
### Sentiment Analysis Model
Our goal is to train, optimize and finetune a Text Classification model using a [pre-trained model](https://huggingface.co/distilbert/distilbert-base-uncased) by using [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) movie reviews dataset. 

The data preprocessing can be seen in [1-preprocessing](https://github.com/tashrifmahmud/Sentiment-Analysis-Model/blob/main/notebooks/1-preprocessing.ipynb) notebook.

We also try traditional scikit-learn models with our preprocessed dateset. Details of it can be found in [2-representation](https://github.com/tashrifmahmud/Sentiment-Analysis-Model/blob/main/notebooks/2-representation.ipynb) notebook. 

By using Distilbert, we train our initial Sentiment Analysis model. The process can be found in [3-pre-trained-model](https://github.com/tashrifmahmud/Sentiment-Analysis-Model/blob/main/notebooks/3-pre-trained-model.ipynb) notebook.

Our final optimized [model](https://huggingface.co/tashrifmahmud/sentiment_analysis_model_v2) which can be found in [4-optimization](https://github.com/tashrifmahmud/Sentiment-Analysis-Model/blob/main/notebooks/4-optimization.ipynb) notebook is a fine-tuned version of the DistilBERT transformer architecture for sentiment analysis. The model has been further fine-tuned on the [Rotten Tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes) dataset to improve its generalization and performance on movie-related text.

In [5-deployment](https://github.com/tashrifmahmud/Sentiment-Analysis-Model/blob/main/notebooks/5-deployment.ipynb) notebook we evaluate all 3 of our models against a finetuned text classification [model from HuggingFace](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english). 

We also built a [Streamlit App](https://sentiment-analysis-v2.streamlit.app/) and details of it can be found in [app.py](https://github.com/tashrifmahmud/Sentiment-Analysis-Model/blob/main/app.py) script. [Reddit Post Analysis Tool](https://sentiment-analysis-v2.streamlit.app/Reddit_Sentiment_Analysis_Tool) has been added in later revision of this project.

## Dataset
We primarily used [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) dataset to build our own Sentiment Analysis model as well as train and finetune our pretrained [Distilbert](https://huggingface.co/distilbert/distilbert-base-uncased) model.

Afterwards we used [Rotten Tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes) dataset to use transfer learning and further finetuned our existing [Sentiment Analysis](https://huggingface.co/tashrifmahmud/sentiment_analysis_model_v2) model.


### Dataset Preprocessing Steps:
We have conducted several preprocessing on our dataset before training our models. It can be seen in [1-preprocessing](https://github.com/tashrifmahmud/Sentiment-Analysis-Model/main/notebooks/1-preprocessing.ipynb) notebook. Some steps are continued in remaining notebooks.

- Imported and accessed dataset alongside primary inspection, used pandas dataframe for easier cleaning
- Cleaning Text: Removing punctuation, newline characters and trailing spaces + lowercasing characters
- Tokenization: Custom tokenization for own model, used tokenizer from huggingface for pre-trained models 
- Stop Word Removal: Filter out unnecessary words using nltk stopwords english corpus for own model
- Word2Vec: Used Word2Vec to generate embeddings for the cleaned tokens for own model

For pre-trained models like [Distilbert](https://huggingface.co/distilbert/distilbert-base-uncased), used the model's tokenizer and padding methods.

## Custom-Trained Models from Scratch
We built our own Sentiment Analysis models using traditional machine learning techniques before experimenting with pre-trained models.

We evaluated three scikit-learn models:
1. Logistic Regression
2. Random Forest
3. XGBoost

### Custom-Trained Model Performance Metrics

| **Metric**                          | **Logistic Regression** | **Random Forest** | **XGBoost** |
|---------------------------------|---------------------|---------------|---------|
| **Validation Accuracy**         | 0.8056             | 0.7856        | 0.7992  |
| **Precision (Class 0)**         | 0.81               | 0.80          | 0.81    |
| **Precision (Class 1)**         | 0.80               | 0.78          | 0.79    |
| **Recall (Class 0)**            | 0.79               | 0.77          | 0.79    |
| **Recall (Class 1)**            | 0.82               | 0.80          | 0.81    |
| **F1-Score (Class 0)**          | 0.80               | 0.78          | 0.80    |
| **F1-Score (Class 1)**          | 0.81               | 0.79          | 0.80    |
| **Confusion Matrix (Class 0)**  | 1997 True, 518 False | 1943 True, 572 False | 1978 True, 537 False |
| **Confusion Matrix (Class 1)**  | 2031 True, 454 False | 1985 True, 500 False | 2018 True, 467 False |

The Logistic Regression model achieved the highest validation accuracy of 80.56%, followed by XGBoost at 79.92% and Random Forest at 78.56%. All models performed similarly in terms of precision, recall, and F1-score, with slight variations between them. Logistic Regression showed the best overall performance, while XGBoost had the most balanced results across both classes.

## Pre-Trained Model
We used [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) pre-trained model to train our text classification model. We used [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) dataset and preprocessed it with [distilbert's tokenizer](https://huggingface.co/distilbert/distilbert-base-uncased/blob/main/tokenizer.json) and [Data Collator](https://huggingface.co/docs/transformers/en/main_classes/data_collator) padding.

**Evaluation Result 1:**

| Epoch | Training Loss | Validation Loss | Accuracy |
|-------|---------------|-----------------|----------|
| 1     | 0.221500      | 0.207077        | 0.921280 |
| 2     | 0.145500      | 0.236552        | 0.931040 |

## Pre-Trained Model Optimization
We further optimized our intial [Sentiment Analysis](https://huggingface.co/tashrifmahmud/sentiment_analysis_model) by finetuning it with Transfer Learning method. We used [Rotten Tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes) dataset to train, by freezing the initial layers(except for the classification head) and fine-tune only the later layers. This helped us in retaining the knowledge learned from [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) and focuses the model on learning the specific features of [Rotten Tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes).

> Final Model: [Model](https://huggingface.co/tashrifmahmud/sentiment_analysis_model_v2) | [App](https://sentiment-analysis-v2.streamlit.app/)

**Evaluation Result 2:**
| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall  | F1      |
|-------|---------------|-----------------|----------|-----------|---------|---------|
| 1     | 0.365000      | 0.368247        | 0.839587 | 0.826715  | 0.859287| 0.842686|
| 2     | 0.280400      | 0.389193        | 0.845216 | 0.852490  | 0.834897| 0.843602|
| 3     | 0.230100      | 0.434206        | 0.844278 | 0.840445  | 0.849906| 0.845149|

It appears that the model starts to overfit after the first two epochs, as evidenced by the increasing validation loss—from 0.37 to 0.39, and ultimately to 0.43 in the final epoch. To prevent overfitting, we will use the model checkpoint prior to this increase in validation loss.

## Model Testing

We tested a random sample of 1000 unseen reviews from Rotten Tomatoes to see how our intial pre-trained [Sentiment Analysis Model](https://huggingface.co/tashrifmahmud/sentiment_analysis_model) and optimized [Sentiment Analysis Model v2](https://huggingface.co/tashrifmahmud/sentiment_analysis_model_v2) performs head-to-head.

| Metric                     | Initial Model             | Optimized Model            |
|----------------------------|-----------------------|----------------------|
| **Loss**                   | 0.4268                | 0.3621               |
| **Accuracy**               | 82.1%                 | 84.3%                |
| **Precision**              | 81.61%                | 84.81%               |
| **Recall**                 | 85.74%                | 85.93%               |
| **F1 Score**               | 83.62%                | 85.37%               |

The optimized model shows notable improvements: accuracy increased from 82.1% to 84.3%, precision from 81.61% to 84.81%, and F1 score from 83.62% to 85.37%. Loss decreased from 0.4268 to 0.3621, runtime dropped from 17.7 to 15.61 seconds, and throughput improved with samples per second rising from 56.49 to 64.05, and steps per second from 7.06 to 8.01.

We also tested 3 random sample reviews to see how our [Logistic Regression Model](https://github.com/tashrifmahmud/Sentiment-Analysis-Model/blob/main/notebooks/2-representation.ipynb), intial pre-trained [Sentiment Analysis Model](https://huggingface.co/tashrifmahmud/sentiment_analysis_model),optimized [Sentiment Analysis Model v2](https://huggingface.co/tashrifmahmud/sentiment_analysis_model_v2) and a dedicated and popular finetuned [Distilber](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) model fares when put to the challenge! 

| Movie Review         | [Logistic Regression](https://github.com/tashrifmahmud/Sentiment-Analysis-Model/blob/main/notebooks/2-representation.ipynb) | [Initial Model](https://huggingface.co/tashrifmahmud/sentiment_analysis_model) | [Optimized Model](https://huggingface.co/tashrifmahmud/sentiment_analysis_model_v2) | [Distilbert Model](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) |
|---------------------|---------------------|-------------|-------------|------------|
| Sample 1            | 0.7086              | 0.9567      | 0.9684      | 0.9999     |
| Sample 2            | 0.8844              | 0.9699      | 0.9619      | 0.9978     |
| Sample 3            | 0.7940              | 0.9904      | 0.9844      | 0.9995     |

Without a doubt the finetuned [Distilbert Model](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) model performs the best, reason why it is the most popular in Hugging Face's Text Classification models. Our intial and optimized pre-trained model also performs relatively well. But Logistic Regression Model cannot keep up in terms of score and might make mistakes with more unseen review testing.

## Performance Metrics of Final Model
Optimized [Sentiment Analysis Model v2](https://huggingface.co/tashrifmahmud/sentiment_analysis_model_v2) model:
- Loss: 0.3682
- Accuracy: 0.8396
- Precision: 0.8267
- Recall: 0.8593
- F1: 0.8427

## Hyperparameters
The following hyperparameters were used during training:

- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 3

The model starts to overfit after after 2 epochs and best model is determined at epoch 1.0 (checkpoint-534). Hyperparameters can be tuned further to achieve better results but could not be done due to time and resource limitations.  

## Links
🤗: [Hugging Face](https://huggingface.co/tashrifmahmud/sentiment_analysis_model_v2) | :bar_chart: [Streamlit App](https://sentiment-analysis-v2.streamlit.app/) | :film_strip: [YouTube Video](https://www.youtube.com/watch?v=a1PKN0u6dso)

> Tech Stack: Python | Pandas | NumPy | Matplotlib | Scikit-Learn | NLTK | Hugging Face Transformers | NLP (TF-IDF, Sentence Transformers) | Streamlit | Kaggle

