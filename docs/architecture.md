# System Architecture: Fake News Detection Application

## 1. Executive Summary

The **Fake News Detection System** is structured using the **Model-View-Presenter (MVP)** software architecture pattern. This architectural choice enforces strict separation of concerns, decoupling Machine Learning model training and inference logic from HTTP routing, request validation, and user interface rendering.

---

## 2. High-Level System Architecture

```mermaid
flowchart TD
    subgraph Client["Client (Web Browser)"]
        UI["User Interface (HTML5/CSS3/JS)"]
    end

    subgraph PresenterLayer["Presenter Layer (Flask Controllers & Validation)"]
        App["app.py (Flask Factory & Middleware)"]
        Routes["presenter/routes.py (Blueprint Endpoints)"]
        Presenter["presenter/presenter.py (Business Logic Mediator)"]
        Validation["presenter/validation.py (Input Validation)"]
    end

    subgraph ModelLayer["Model Layer (ML Inference & Data Pipeline)"]
        Predictor["model/predictor.py (Inference Engine)"]
        Preprocessing["model/preprocessing.py (Text Cleaning & NLP)"]
        SaveModel["model/save_model.py (Joblib Serialization)"]
        TrainModel["model/train_model.py (ML Training Pipeline)"]
        Artifacts[("model.pkl & vectorizer.pkl Artifacts")]
    end

    UI -->|HTTP GET / POST| Routes
    App -->|Registers| Routes
    Routes -->|Delegates Request| Presenter
    Presenter -->|Validates Input| Validation
    Presenter -->|Calls predict_news| Predictor
    Predictor -->|Cleans Text| Preprocessing
    Predictor -->|Loads Artifacts| SaveModel
    SaveModel -->|Reads| Artifacts
    Predictor -->|Calculates Probabilities| Presenter
    Presenter -->|Formats View Payload| Routes
    Routes -->|Renders HTML / JSON| UI
```

---

## 3. Data Flow Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as User Browser
    participant Routes as presenter/routes.py
    participant Presenter as presenter/presenter.py
    participant Validator as presenter/validation.py
    participant Predictor as model/predictor.py
    participant Preprocessing as model/preprocessing.py
    participant Artifacts as Joblib (model.pkl / vectorizer.pkl)

    User->>Routes: POST /predict (news_text="...")
    Routes->>Presenter: predict_news(news_text)
    Presenter->>Validator: validate_news_input(news_text)
    alt Invalid Input (Empty / Whitespace)
        Validator-->>Presenter: (False, "Article text cannot be empty.")
        Presenter-->>Routes: { success: False, error: "..." }
        Routes-->>User: Render index.html (Status 400 + Error Alert)
    else Valid Input
        Validator-->>Presenter: (True, "")
        Presenter->>Predictor: model_predict_news(news_text)
        Predictor->>Preprocessing: clean_text(news_text)
        Preprocessing-->>Predictor: cleaned_text
        Predictor->>Artifacts: load_model() & load_vectorizer()
        Artifacts-->>Predictor: Model & Vectorizer objects
        Predictor->>Predictor: vectorizer.transform([cleaned_text])
        Predictor->>Predictor: model.predict_proba() & model.predict()
        Predictor-->>Presenter: { prediction: "REAL", label: 1, confidence: 98.65 }
        Presenter->>Presenter: format_prediction_result(raw_result)
        Presenter-->>Routes: { success: True, result: formatted_payload }
        Routes-->>User: Render result.html (Status 200 + Badge & Progress Meter)
    end
```

---

## 4. MVP Architecture & Layer Responsibilities

### 4.1. Model Layer (`model/`)
- **Isolation**: Pure Python code with no dependencies on Flask or web request contexts.
- **`model/preprocessing.py`**: Executes NLP text cleaning (stripping special characters, removing extra whitespace, lowercasing).
- **`model/train_model.py`**: Trains Logistic Regression classifier on `dataset/cleaned_news.csv`, logs environment package versions, and serializes trained objects.
- **`model/save_model.py`**: Utility functions wrapping `joblib.dump` and `joblib.load` with robust error messaging.
- **`model/predictor.py`**: Singleton artifact loader and inference module computing prediction classes and confidence percentages.

### 4.2. Presenter Layer (`presenter/`)
- **Mediator Role**: Serves as the middle layer between View and Model.
- **`presenter/routes.py`**: Defines Flask Blueprint `presenter_bp` for endpoints (`/`, `/predict`, `/about`, `/health`).
- **`presenter/validation.py`**: Implements input validation rules checking character lengths and non-empty strings.
- **`presenter/presenter.py`**: Coordinates request validation, invokes Model predictor, and formats structured payloads for template rendering.

### 4.3. View Layer (`view/`)
- **Passive View**: Displays UI elements and forwards user form submissions. Contains zero business logic or ML model references.
- **`view/templates/`**: Jinja2 templates (`base.html`, `index.html`, `result.html`).
- **`view/static/`**: Custom CSS styling (`style.css`) and client-side interactivity (`script.js`).

---

## 5. Architectural Benefits

1. **Testability**: Presenter logic and Model layer can be unit-tested independently without mocking HTTP request contexts.
2. **Resilience**: Model loading errors or scikit-learn version mismatches are caught gracefully in `predictor.py` and returned as readable error messages instead of causing unhandled web server crashes.
3. **Maintainability**: Swapping out ML algorithms (e.g. replacing Logistic Regression with BERT) only requires updating `model/predictor.py` while leaving Presenter and View layers completely unchanged.
