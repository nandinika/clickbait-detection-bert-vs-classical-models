# A Comparative Study of Classical and BERT-based Models for Clickbait Detection

This project explores the task of **clickbait headline detection** using both traditional machine learning and transformer-based NLP models.  
It compares the performance of classical algorithms (Logistic Regression, Naive Bayes) with **BERT**, highlighting the strengths of contextual embeddings in detecting misleading or sensationalized news headlines.

---

## Overview
- **Goal:** Detect clickbait headlines in news-related Reddit posts.
- **Dataset:** 3,000+ posts scraped from multiple subreddits (e.g., r/worldnews, r/clickbait, r/politics).
- **Labeling:** Heuristic rules based on capitalization, punctuation, and linguistic cues.
- **Models Used:** Logistic Regression, Naive bayes, and BERT.
- **Performance:** Achieved an **F1-score of 0.90** using BERT-based transformer.

---

## Project Workflow
1. **Data Collection:**  
   Scraped Reddit posts using the **Reddit API (PRAW)**.
2. **Data Preprocessing:**  
   Tokenization, stopword removal, and normalization.
3. **Heuristic Labeling:**  
   Classified posts as clickbait or non-clickbait using text-based rules.
4. **Model Training:**  
   - Classical models: Logistic Regression, SVM, Random Forest.  
   - Deep model: Fine-tuned **BERT (Hugging Face Transformers)**.
5. **Evaluation:**  
   Compared models on accuracy, F1-score, and interpretability.
6. **Visualization:**  
   Used **Matplotlib** and **Seaborn** to visualize engagement trends and linguistic patterns.

---

## Tech Stack
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Transformers, Matplotlib, Seaborn, NLTK  
- **APIs:** Reddit API (PRAW)

---

## Results
| Model | Accuracy | F1-Score |
|--------|-----------|----------|
| Logistic Regression | 0.84 | 0.84 |
| Naive Bayes  | 0.85 | 0.85 |
| BERT | **0.90** | **0.90** |

