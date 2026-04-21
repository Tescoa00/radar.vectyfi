# 🎯 Week 1: Baseline & API Foundation

**Sprint Goal:** Move out of the Jupyter Notebook environment. By Friday, we need to have our backend architecture set up, our baseline model exported, and the API containerized in Docker.

---

## 🛑 Milestone 1: "The Skeleton" (Deadline: Wednesday EOD)

### [ ] W1-01: Baseline Model & Pipeline Export
* **Focus:** Data Science
* **Goal:** Train the first working baseline model and export it locally.
* **Tasks:**
  * [ ] Train a baseline model in the notebook (the score doesn't matter right now!).
  * [ ] Define the preprocessing pipeline (scalers, encoders).
  * [ ] Export the model and pipeline as a `.pkl` or `.joblib` file locally.
* *Note: Be pragmatic. Do not package all your EDA. Only the code that transforms raw data into a prediction matters.*

### [ ] W1-02: Backend Repo Setup & Structuring
* **Focus:** Data Engineering / GitHub
* **Goal:** Set up a clean backend repository.
* **Tasks:**
  * [ ] Create the GitHub repo (Reference: `julesvanrie/base_project_back`).
  * [ ] Add a strict `.gitignore` file to keep datasets and `.env` files out of source control.
  * [ ] Create the folder structure (e.g., `api/`, `ml_logic/`, `notebooks/`).
  * [ ] Invite all team members as collaborators.

### [ ] W1-03: Dummy API Endpoint
* **Focus:** Backend / FastAPI
* **Goal:** The API must be able to respond to requests, even if the real model isn't attached yet.
* **Tasks:**
  * [ ] Initialize FastAPI.
  * [ ] Create a `/predict` endpoint.
  * [ ] Return a hardcoded dummy response (e.g., `{"prediction": "awarded", "confidence": 0.85}`).
  * [ ] Test the API locally using Uvicorn.

---

## 🚀 Milestone 2: "The Engine" (Deadline: Friday EOD)

### [ ] W1-04: API Package Integration
* **Focus:** Backend / Data Science
* **Goal:** Connect the dummy API to the real model exported on Wednesday.
* **Tasks:**
  * [ ] Write a `load_model()` function to load the `.pkl`/`.joblib` file.
  * [ ] Write a `preprocess()` function to pass user input through the pipeline.
  * [ ] Update the `/predict` endpoint to use these functions and return a real prediction.

### [ ] W1-05: Dockerization (Local Image)
* **Focus:** DevOps / Infrastructure
* **Goal:** Package the API into a machine-independent container.
* **Tasks:**
  * [ ] Write the `Dockerfile` (Python base image, install requirements, start Uvicorn).
  * [ ] Define a clean `requirements.txt`.
  * [ ] Build the Docker image locally (`docker build`).
  * [ ] Run the container locally and successfully test the `/predict` endpoint (`docker run`).

### [ ] W1-06: Cloud Deployment (Bonus Target)
* **Focus:** DevOps
* **Goal:** Bring the API online.
* **Tasks:**
  * [ ] Push the Docker image to a Container Registry (e.g., Google Artifact Registry).
  * [ ] Deploy the image on a cloud service (e.g., Google Cloud Run).
  * [ ] Hand over the live URL to the front-end team.