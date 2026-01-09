# Startup Guide

Clone this repository.

Set up a .env file in the project root.

[Example of .env](./.env.txt)

Have docker started and from the project root:

```
docker compose up --build -d
```

# Infrastructure Entry Points

You can access 

## MinIO
```
http://localhost:{$MINIO_PORT}
```

## MLflow UI
```
http://localhost:${MLFLOW_PORT}
```

## Airflow UI
```
http://localhost:8080
```

# Inference API

Swagger UI

```
http://localhost:${MODELAPI_PORT}/swagger
```

Main endpoint 

```
POST /api
```

Example Request

```
curl -X POST http://localhost:8000/api \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a sample message"
  }'
```

Responce

```
{
  "confidence_score": 0.5631636113229244,
  "prediction": "offensive language",
  "prediction_index": 1,
  "text": "This is a sample message"
}
```

0 -> hate speech 

1 -> offensive language

2 -> neither

## Training

On service start model is trained on last processed data and results are loged in MLflow.

## Airflow

Airflow is used for making dataset with last processed data. It uses inference logs and loaded moderator info.

[DAGs location.](./airflow/dags)


# Shut down

```
docker compose down
```

To shut down with data delition.

```
docker compose down -v
```