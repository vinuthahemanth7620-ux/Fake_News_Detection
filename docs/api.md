# API Endpoint Documentation

## 1. Web Page Routes

### `GET /`
- **Description**: Renders the main Fake News Detector homepage view (`index.html`).
- **Response**: HTML Page.

### `POST /predict`
- **Description**: Accepts news article text and returns classification result page (`result.html`).
- **Form Parameters**:
  - `article_text` (string, required): News article text content.

## 2. REST API Endpoints (Optional / AJAX)

### `POST /api/predict`
- **Description**: JSON endpoint for asynchronous classification.
- **Request Body**:
  ```json
  {
    "text": "Breaking news article content..."
  }
  ```
- **Response Payload**:
  ```json
  {
    "status": "success",
    "prediction": "REAL",
    "confidence": 0.942,
    "processed_text_sample": "breaking news article content..."
  }
  ```
