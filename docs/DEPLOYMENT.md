# Cloud Deployment Guide: Render.com & Vercel

This repository includes a native **Render Blueprint (`render.yaml`)** that allows you to deploy the entire production stack (FastAPI Backend, PostgreSQL DB, Redis Velocity Cache, and React Frontend) in **one click**.

---

## Architecture Overview

```
                      ┌──────────────────────────┐
                      │  React + Vite Frontend   │
                      │  (Render Static Site or  │
                      │      Vercel / Netlify)   │
                      └────────────┬─────────────┘
                                   │ HTTPS / REST (X-API-Key)
                                   ▼
                      ┌──────────────────────────┐
                      │    FastAPI Backend API   │
                      │   (Render Web Service)   │
                      └───────┬──────────┬───────┘
                              │          │
                     Sub-2ms  │          │ WAL / Relational
                     Velocity │          │ Storage
                              ▼          ▼
                      ┌───────────────┐ ┌──────────────────┐
                      │  Redis Cache  │ │  PostgreSQL DB   │
                      └───────────────┘ └──────────────────┘
```

---

## Option 1: Automatic 1-Click Deployment with Render Blueprint (Recommended)

Render can parse `render.yaml` in the root of your repository and provision:
1. **`fraud-scoring-api`**: Python Web Service running FastAPI (`/healthz` health check).
2. **`fraud-scoring-frontend`**: Static Web Site hosting the compiled React application.
3. **`fraud-db`**: Managed PostgreSQL database.
4. **`fraud-redis`**: Managed in-memory Redis instance for real-time velocity calculations.

### Step-by-Step Instructions:
1. **Push your code to GitHub**:
   ```bash
   git push origin main
   ```
2. Log into [Render.com](https://render.com).
3. Click **New +** in the top navigation bar and select **Blueprint**.
4. Connect your GitHub repository: `AshokSingodia-Codes/fraud-scoring-service`.
5. Render will detect `render.yaml` and display the 4 services to create.
6. Click **Apply**.
7. Render will automatically build the backend, provision the databases, compile the frontend, and give you live public URLs!

---

## Option 2: Deploy Frontend on Vercel + Backend on Render

If you prefer using **Vercel** for the frontend:

### A. Deploy Backend on Render:
1. In [Render.com](https://render.com), click **New +** -> **Web Service**.
2. Connect your repo and configure:
   - **Root Directory**: `.`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.fraud.serving.app:app --host 0.0.0.0 --port $PORT --workers 2`
3. Add Environment Variables:
   - `APP_ENV`: `prod`
   - `ALLOWED_API_KEYS`: `your-production-secret-api-key`
   - `DATABASE_URL`: Your PostgreSQL connection string (or leave default for local SQLite).
   - `REDIS_URL`: Your Redis connection string.
4. Note your backend URL (e.g., `https://fraud-scoring-api.onrender.com`).

### B. Deploy Frontend on Vercel:
1. In [Vercel.com](https://vercel.com), click **Add New...** -> **Project**.
2. Select your repository `AshokSingodia-Codes/fraud-scoring-service`.
3. Set **Root Directory** to `frontend`.
4. Framework Preset: **Vite**.
5. Add Environment Variables:
   - `VITE_API_URL`: `https://fraud-scoring-api.onrender.com`
   - `VITE_API_KEY`: `your-production-secret-api-key`
6. Click **Deploy**.

---

## Verifying the Deployment

Once deployed:
1. Open your live frontend URL.
2. Click **Single Check** -> click **Populate Typical Legitimate** -> click **Analyze Transaction Risk**.
3. Confirm you receive a live response with SHAP factors and decision within milliseconds.
4. Visit `https://<your-backend-url>/docs` to view the interactive OpenAPI documentation.
