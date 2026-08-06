# API & Endpoint Documentation: Fake News Detection Application

This document provides specifications for all web endpoints and health check APIs available in the Fake News Detection application.

---

## Endpoint Overview

| Endpoint | Method | Content-Type | Description |
|---|---|---|---|
| `/` | `GET` | `text/html` | Renders main home view with news text submission form |
| `/predict` | `POST` | `application/x-www-form-urlencoded` or `application/json` | Classifies input text and renders result page or returns JSON |
| `/about` | `GET` | `text/html` | Displays project information page |
| `/health` | `GET` | `application/json` | Returns system health and model artifact loading status |

---

## 1. `GET /`

Renders the home page view containing the news article text input form, character counter, and sample article test buttons.

### Request Example
```http
GET / HTTP/1.1
Host: 127.0.0.1:5000
Accept: text/html
```

### Response Example
- **Status**: `200 OK`
- **Content-Type**: `text/html; charset=utf-8`
- **Body**: Rendered `index.html` page.

---

## 2. `POST /predict`

Accepts raw news article text from form data or JSON payload, validates input, calls `presenter.predict_news()`, and renders `result.html` with classification badges and confidence progress meters.

### Request Example (Form Submission)
```http
POST /predict HTTP/1.1
Host: 127.0.0.1:5000
Content-Type: application/x-www-form-urlencoded

news_text=The+Reserve+Bank+of+India+announced+that+it+will+maintain+the+repo+rate+unchanged.
```

### Request Example (JSON Submission)
```http
POST /predict HTTP/1.1
Host: 127.0.0.1:5000
Content-Type: application/json

{
  "news_text": "The Reserve Bank of India announced that it will maintain the repo rate unchanged."
}
```

### Success Response Example (Status 200)
- **Status**: `200 OK`
- **Content-Type**: `text/html; charset=utf-8`
- **Body**: Rendered `result.html` page displaying:
  - Prediction: `REAL` or `FAKE`
  - Confidence Percentage: e.g., `98.65%`
  - Badge Styling Class: `success` (Real) or `danger` (Fake)
  - Submitted Article Excerpt.

### Validation / Error Response Example (Status 400)
Occurs when submitted text is empty or whitespace-only.
- **Status**: `400 Bad Request`
- **Content-Type**: `text/html; charset=utf-8`
- **Body**: Rendered `index.html` page containing Bootstrap warning alert:
  ```html
  <div class="alert alert-danger">
      <strong>Validation Error:</strong> Article text cannot be empty.
  </div>
  ```

---

## 3. `GET /about`

Displays information about the Fake News Detector architecture, machine learning model, and technologies used.

### Request Example
```http
GET /about HTTP/1.1
Host: 127.0.0.1:5000
Accept: text/html
```

### Response Example
- **Status**: `200 OK`
- **Content-Type**: `text/html; charset=utf-8`
- **Body**: Rendered project information view.

---

## 4. `GET /health`

Automated health check endpoint used by monitoring services and container orchestrators to verify application availability and ML model artifact persistence.

### Request Example
```http
GET /health HTTP/1.1
Host: 127.0.0.1:5000
Accept: application/json
```

### Healthy Response Example (Status 200)
Returned when both `model/model.pkl` and `model/vectorizer.pkl` exist and are accessible.
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Unhealthy Response Example (Status 503)
Returned if model or vectorizer artifacts are missing or unreadable.
```json
{
  "status": "unhealthy",
  "model_loaded": false
}
```
