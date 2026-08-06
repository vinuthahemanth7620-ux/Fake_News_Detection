# Fake News Detection System using Machine Learning & Flask

An end-to-end Machine Learning web application designed to classify news articles as **REAL** or **FAKE** in real time using Natural Language Processing (NLP) and an optimized **Linear Support Vector Classifier (LinearSVC)**, built cleanly following the **Model-View-Presenter (MVP)** architecture pattern.

---

## 🌟 Project Overview

Misinformation and fake news pose significant threats to public trust and information integrity. The **Fake News Detection System** leverages automated NLP feature extraction (optimized TF-IDF Vectorization) and LinearSVC classification to analyze textual features of news articles and output confidence-scored predictions (**REAL** or **FAKE**).

The application is architected according to the **Model-View-Presenter (MVP)** software pattern, ensuring complete decoupling between Machine Learning model execution, business/validation logic, and the user interface.

---

## ✨ Features

- **Real-Time Text Classification**: Classifies news article headlines or bodies as REAL or FAKE with high precision (99.62% accuracy).
- **Confidence Scoring**: Displays probability confidence percentages for every prediction.
- **Interactive UI**: Clean, responsive interface built with Bootstrap 5, custom CSS styling, and real-time character counting.
- **Client-Side & Server-Side Validation**: Ensures input validity before running inference.
- **Sample Article Quick-Testing**: Allows 1-click loading of sample Real and Fake news articles for instant verification.
- **System Health Endpoint**: Provides a JSON health check endpoint (`/health`) verifying system health and model artifact loading.
- **Comprehensive Logging & Artifacts**: Full request-level logging, cross-validation exports, and confusion matrix visualizations.

---

## 🛠️ Technologies Used

| Category | Technology |
|---|---|
| **Programming Language** | Python 3.13+ |
| **Machine Learning / Data Processing** | scikit-learn (1.7.2), pandas (2.3.3), numpy (2.3.5), joblib (1.5.2), nltk (3.10.1), matplotlib (3.10.0) |
| **Web Framework** | Flask (3.1.2) |
| **Frontend Layout & Styling** | HTML5, CSS3, Bootstrap 5.3, Bootstrap Icons, Google Fonts (Inter) |
| **Client Scripting** | Vanilla JavaScript (ES6+) |
| **Testing** | pytest (9.0.2) |

---

## 📐 Project Architecture (MVP Pattern)

The application follows the strict **Model-View-Presenter (MVP)** architectural design pattern:

```text
[ View Layer ] <------ HTTP Requests / Responses ------> [ Presenter Layer ] <------ Data / Inferences ------> [ Model Layer ]
 (HTML5 / JS / CSS)                                      (routes.py, presenter.py)                             (predictor, train_model)
```

- **Model Layer (`model/`)**: Pure Machine Learning module responsible for dataset loading, TF-IDF vectorization, model training, artifact serialization, and text inference. Completely isolated from Flask and UI templates.
- **Presenter Layer (`presenter/`)**: Contains Flask Blueprints, route handlers (`routes.py`), validation logic (`validation.py`), and presenter orchestration (`presenter.py`). Receives user requests, delegates to Model layer, and formats data payloads for rendering.
- **View Layer (`view/`)**: HTML5 templates (`view/templates/`) and static assets (`view/static/`) for UI rendering. Views never communicate directly with Model objects.

---

## 📁 Folder Structure

```text
fake news detection/
├── app.py                      # Main Flask application entry point & factory
├── config.py                   # Central application configuration
├── requirements.txt            # Locked exact dependency package versions
├── README.md                   # Project documentation & usage guide
├── dataset/
│   └── cleaned_news.csv        # Preprocessed dataset (44,182 news records)
├── artifacts/                  # Generated evaluation & comparison artifacts
│   ├── feature_comparison.csv       # TF-IDF parameter grid search comparison
│   ├── cross_validation_results.csv # 5-fold Stratified CV scores for all configs
│   ├── model_metrics.json           # JSON metrics summary & top TF-IDF features
│   └── confusion_matrix.png         # Confusion matrix plot for winning model
├── model/                      # Model Layer
│   ├── __init__.py
│   ├── train_model.py          # ML training & feature optimization pipeline
│   ├── save_model.py           # Joblib artifact serialization / deserialization helpers
│   ├── predictor.py            # Article inference module & exception handling
│   ├── preprocessing.py        # Text cleaning & NLP preprocessing pipeline
│   ├── model.pkl               # Serialized trained LinearSVC model
│   └── vectorizer.pkl          # Serialized fitted TF-IDF Vectorizer
├── presenter/                  # Presenter Layer
│   ├── __init__.py
│   ├── routes.py               # Flask Blueprint HTTP endpoint definitions
│   ├── presenter.py            # Presenter business logic & response formatting
│   └── validation.py           # Input text validation rules
├── view/                       # View Layer
│   ├── templates/
│   │   ├── base.html           # Base layout template with navbar & footer
│   │   ├── index.html          # Main home view with text input form & sample pills
│   │   └── result.html         # Prediction result view with confidence meter
│   └── static/
│       ├── css/
│       │   └── style.css       # Custom design system & animations
│       └── js/
│           └── script.js       # Client-side validation, char counter & spinner
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── logger.py               # Application logger configuration
│   └── helper.py               # File verification and directory utilities
├── tests/                      # Unit & Integration Test Suite
│   ├── test_prediction.py      # Predictor unit tests
│   ├── test_preprocessing.py   # Preprocessing unit tests
│   └── test_routes.py          # Route integration tests
└── docs/                       # Project Documentation
    ├── architecture.md         # Detailed architectural documentation
    └── api.md                  # REST & HTTP API endpoint specifications
```

---

## 📊 Dataset Information

The model is trained on a comprehensive dataset of **44,182 news records** (`dataset/cleaned_news.csv`), combining curated news articles labeled as Real or Fake.

- **Total Records**: 44,182
- **Training Samples**: 35,345 (80%)
- **Testing Samples**: 8,837 (20%)
- **Target Labels**: `1` (REAL) or `0` (FAKE)

---

## 🎯 Best Model & Feature Configuration

Through multi-model evaluation and 36-configuration TF-IDF hyperparameter optimization, **LinearSVC** paired with unigram/bigram TF-IDF vectorization achieved optimal generalization without overfitting.

### 🏆 Winning Configuration: `Config_31`
- **Best Classifier**: Linear Support Vector Classifier (`LinearSVC(C=1.0, max_iter=2000, dual='auto')`)
- **Max Features (`max_features`)**: `20,000`
- **N-Gram Range (`ngram_range`)**: `(1, 2)` (Unigrams & Bigrams)
- **Minimum Document Frequency (`min_df`)**: `5`
- **Maximum Document Frequency (`max_df`)**: `0.9`
- **5-Fold Stratified CV Mean F1-Score**: **`0.9940 ± 0.0003`**

---

## 📈 Final Evaluation Metrics

Evaluated on the 8,837 independent test samples (stratified 20% holdout):

| Evaluation Metric | Score |
|---|---|
| **Accuracy** | **99.62%** (`0.9962`) |
| **Precision** | **99.58%** (`0.9958`) |
| **Recall** | **99.63%** (`0.9963`) |
| **F1-Score** | **99.60%** (`0.9960`) |
| **5-Fold CV Mean F1** | **99.40%** (`0.9940`) |

```text
Confusion Matrix:
[[4536   18]   (Fake: 4536 true negatives, 18 false positives)
 [  16 4267]]  (Real: 16 false negatives, 4267 true positives)
```

---

## 🧪 TF-IDF Feature Engineering Comparison (Sample)

| Config ID | max_features | ngram_range | min_df | max_df | 5-Fold CV Mean F1 | Test Accuracy | Test F1-Score |
|---|---|---|---|---|---|---|---|
| **Config_31** | **20,000** | **(1, 2)** | **5** | **0.9** | **0.9940** | **0.9962** | **0.9960** |
| Config_32 | 20,000 | (1, 2) | 5 | 0.95 | 0.9940 | 0.9962 | 0.9960 |
| Config_29 | 20,000 | (1, 2) | 2 | 0.9 | 0.9940 | 0.9960 | 0.9959 |
| Config_33 | 20,000 | (1, 3) | 2 | 0.9 | 0.9938 | 0.9965 | 0.9964 |
| Config_18 | 10,000 | (1, 2) | 2 | 0.95 | 0.9931 | 0.9954 | 0.9952 |
| Config_01 | 5,000 | (1, 1) | 2 | 0.9 | 0.9913 | 0.9934 | 0.9932 |

*(Full grid search results exported to [artifacts/feature_comparison.csv](artifacts/feature_comparison.csv) and [artifacts/cross_validation_results.csv](artifacts/cross_validation_results.csv)).*

---

## 🚀 Installation & Virtual Environment Setup

### 1. Prerequisites
- Python 3.13 or higher installed on your system.
- `git` version control.

### 2. Clone Repository
```bash
git clone https://github.com/vinuthahemanth7620-ux/Fake_News_Detection.git
cd Fake_News_Detection
```

### 3. Create & Activate Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🤖 How to Run Feature Optimization & Training

To run the feature optimization pipeline across 36 TF-IDF parameter configurations using 5-fold Stratified Cross-Validation:

```bash
python -m model.train_model
```

*Output will log environment package versions, feature evaluation progress, test set metrics, artifact exports to `artifacts/`, and saving of `model/model.pkl` and `model/vectorizer.pkl`.*

---

## 💻 How to Run the Application

Start the Flask development server:

```bash
python app.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:5000/
```

---

## 🔌 API Endpoints Summary

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Renders main home page (`index.html`) |
| `/predict` | `POST` | Processes article text and renders result page (`result.html`) |
| `/about` | `GET` | Displays project information page |
| `/health` | `GET` | Returns JSON status `{ "status": "healthy", "model_loaded": true }` |

*For detailed request/response examples, see [docs/api.md](docs/api.md).*

---

## 🖼️ Screenshots & Artifacts Section

### Confusion Matrix Plot
`![Confusion Matrix](artifacts/confusion_matrix.png)`

### UI Screenshots (Placeholders)
- **Home Page**: `![Home Page](docs/images/screenshot_home.png)`
- **REAL News Result**: `![REAL Result](docs/images/screenshot_result_real.png)`
- **FAKE News Result**: `![FAKE Result](docs/images/screenshot_result_fake.png)`

---

## 🔮 Future Enhancements

- **Deep Learning Model Extensions**: Integrate fine-tuned RoBERTa / DeBERTa models for contextual transformer classification.
- **Browser Extension**: Build a Chrome/Firefox extension for 1-click news verification directly on webpage text selections.
- **URL Scraper**: Allow users to submit news web page URLs directly instead of copying article text.
- **Multilingual Support**: Support non-English news articles.

---

## 👨‍💻 Author

**Vinutha Hemanth**  
- Repository: [Fake_News_Detection GitHub](https://github.com/vinuthahemanth7620-ux/Fake_News_Detection)

---

## 📜 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
