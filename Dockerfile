FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

RUN uv pip install --system -r pyproject.toml fastapi uvicorn python-multipart

COPY app/ ./app/

ENV PYTHONPATH="/app/app"

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "7860"]