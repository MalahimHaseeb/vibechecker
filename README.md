# VibeChecker

![CI](https://github.com/MalahimHaseeb/vibechecker/actions/workflows/ci.yml/badge.svg)

A Chrome extension that reads the comments on a YouTube video and tells you the vibe: how many are positive, neutral and negative. A scikit-learn model served by FastAPI does the classifying, and MLflow tracks every experiment.

![Extension popup](docs/extension-popup.png)

## How it works

```
YouTube page -> content.js (reads loaded comments) -> popup.js -> background.js -> POST /predict (FastAPI) -> TF-IDF + Logistic Regression -> sentiment counts back to the popup
```

- The extension collects the comments currently loaded on the page.
- The background worker sends them in one batch to the API.
- The API cleans the text and predicts negative, neutral or positive for each comment.
- The popup shows the percentage split.

## Model

Trained on about 37k labeled Reddit comments with three classes (negative, neutral, positive).

| Model | Accuracy | Macro F1 |
|---|---|---|
| Baseline (TF-IDF + Logistic Regression, C=1) | 0.8474 | 0.8376 |
| Tuned (Logistic Regression, C=10, word unigrams) | 0.8929 | 0.8832 |

The tuned model came from a 25 trial random search over the vectorizer and classifier settings. Models were selected on a validation split and scored once on a held-out test split. Every trial is logged in MLflow.

![MLflow runs](docs/mlflow-runs.png)

## Quick start

There are two ways to run the API. Both serve it on http://localhost:8000, which is where the extension looks for it.

### Option 1: Docker (no Python needed)

```bash
docker run -p 8000:8000 ghcr.io/malahimhaseeb/vibechecker:latest
```

Check that it is up:

```bash
curl http://localhost:8000/health
```

To build the image yourself instead of pulling it:

```bash
git clone https://github.com/MalahimHaseeb/vibechecker.git
cd vibechecker
docker build -t vibechecker-api .
docker run -p 8000:8000 vibechecker-api
```

### Option 2: Run from source

You need Python 3.10 or newer.

```bash
git clone https://github.com/MalahimHaseeb/vibechecker.git
cd vibechecker
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn api.main:app
```

On Windows activate with `.venv\Scripts\activate`.

### Load the extension

1. Open `chrome://extensions`
2. Turn on Developer mode
3. Click Load unpacked and select the `extension` folder
4. Open a YouTube video, scroll down until comments load, click the VibeChecker icon and press the button

YouTube loads comments as you scroll, so only the comments already on screen are analyzed.

## API

`GET /health` returns `{"status": "ok"}`.

`POST /predict` takes a list of comments and returns a label for each one plus the totals.

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"comments": ["this video is amazing", "worst video ever"]}'
```

```json
{
  "labels": ["positive", "negative"],
  "counts": {"negative": 1, "neutral": 0, "positive": 1},
  "total": 2
}
```

Interactive docs are at http://localhost:8000/docs while the API is running.

## Retrain and track experiments

```bash
python src/experiment.py
mlflow ui
```

Open http://localhost:5000 to compare runs. The script downloads the dataset on first run and caches it in `data/`.

- Set `FIXED_PARAMS = None` in `src/experiment.py` to run the full random search, or fill it in to refit one configuration.
- Run `python src/register_best.py` to register the best run in the MLflow model registry with the `champion` alias.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## CI/CD

- `ci.yml` runs on every pull request: tests, a Docker build with a live health check and prediction call, and a packaged extension zip as a build artifact.
- `publish.yml` runs on every push to `main` and publishes the Docker image to GitHub Container Registry.

## Project structure

```
vibechecker/
  src/
    data.py            dataset download and caching
    preprocess.py      text cleaning shared by training and the API
    train.py           baseline training run
    experiment.py      random search with MLflow tracking
    register_best.py   registers the best run as champion
  api/main.py          FastAPI service
  extension/           Manifest V3 Chrome extension
  models/              trained model
  tests/               pytest suite
  docs/                screenshots
  Dockerfile           API image
  .github/workflows/   CI and image publishing
```

## Limitations

- The model is trained on Reddit text, and YouTube comments use different slang, emojis and shorthand, so accuracy will be lower than the numbers above.
- English only.
- Only comments loaded on the page are analyzed.
- The extension points at `localhost:8000`, so the API has to be running on your machine.

## Roadmap

- Retraining workflow with a metric gate before promoting a new model
- Hosted API with HTTPS and a Chrome Web Store release
- A small hand-labeled YouTube evaluation set

## Author

Malahim Haseeb, [malahim.dev](https://malahim.dev), [GitHub](https://github.com/MalahimHaseeb)

## License

MIT