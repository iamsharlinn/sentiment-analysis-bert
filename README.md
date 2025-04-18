# 📊 Sentiment Analysis with BERT

## 🔍 Project Overview
This project applies a pretrained BERT model to classify the sentiment of customer reviews as **positive** or **negative**. It uses the HuggingFace Transformers library for model loading and prediction, and the Yelp Polarity dataset for testing. Evaluation includes accuracy, precision, recall, F1-score, and a confusion matrix.

---

## 📁 Dataset
- **Source**: [Yelp Polarity Dataset](https://huggingface.co/datasets/yelp_polarity)
- **Size Used**: 100 reviews (for demonstration)
- **Classes**: POSITIVE, NEGATIVE

---

## 🧠 Techniques & Tools
- BERT-based transformer models (HuggingFace pipeline)
- Text preprocessing and truncation handling (512 token limit)
- Sentiment classification using pretrained `bert-base-uncased`
- Performance metrics: accuracy, precision, recall, F1-score
- Visualization: Seaborn & Matplotlib

---

## 📈 Results
- Achieved ~87% accuracy on sample test set
- High precision and recall for both sentiment classes
- Generated a confusion matrix for error analysis

---

## 🛠️ Technologies
- Python
- HuggingFace Transformers
- Datasets Library
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib
- Jupyter Notebook

---

## 🚀 How to Run
1. Clone this repo  
2. Install dependencies:  
   ```bash
   pip install transformers datasets scikit-learn pandas matplotlib seaborn
