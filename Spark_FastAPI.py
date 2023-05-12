import nest_asyncio
from pyngrok import ngrok
import uvicorn

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
from typing import List

from fastapi import FastAPI, HTTPException, Request, Form, Depends
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

# Import your PySpark ML model and preprocessing functions here
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import Imputer
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel

app = FastAPI()

# enable CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# Add the following line to mount static files
# app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def serve_homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


spark = SparkSession.builder \
    .appName("Fetal Health Prediction") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.python.worker.memory", "2g") \
    .getOrCreate()


data = spark.read.csv("/app/data/fetal_health.csv", header=True, inferSchema=True)

# Handle missing values (if any)
imputer = Imputer(strategy='mean', inputCols=data.columns, outputCols=data.columns)
data = imputer.fit(data).transform(data)

# Assemble features into a single vector
feature_columns = data.columns[:-1]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# feature scaling
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Create pipeline for preprocessing
pipeline = Pipeline(stages=[assembler, scaler])

# Fit the pipeline and transform the data
preprocessing_pipeline_model = pipeline.fit(data)
preprocessed_data = preprocessing_pipeline_model.transform(data) #Just got this fixed up


# Split the dataset into training and testing sets
train_data, test_data = preprocessed_data.randomSplit([0.8, 0.2], seed=42)

# Create a RandomForest model
rf = RandomForestClassifier(labelCol="fetal_health", featuresCol="scaled_features", numTrees=100, seed=42)

# Define input data schema
class InputData(BaseModel):
    baseline_value: float
    accelerations: float
    fetal_movement: float
    uterine_contractions: float
    light_decelerations: float
    severe_decelerations: float
    prolongued_decelerations: float
    abnormal_short_term_variability: float
    mean_value_of_short_term_variability: float
    percentage_of_time_with_abnormal_long_term_variability: float
    mean_value_of_long_term_variability: float
    histogram_width: float
    histogram_min: float
    histogram_max: float
    histogram_number_of_peaks: float
    histogram_number_of_zeroes: float
    histogram_mode: float
    histogram_mean: float
    histogram_median: float
    histogram_variance: float
    histogram_tendency: float

trained_model = None
accuracy_result = None

@app.get("/", response_class=HTMLResponse)
async def serve_homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "accuracy": accuracy_result})

@app.post("/train")
async def train_model(request: Request):
    global trained_model, accuracy_result
    # Load the dataset, preprocess, and train the model
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml import Pipeline
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    # Create a RandomForest model
    rf = RandomForestClassifier(labelCol="fetal_health", featuresCol="scaled_features", numTrees=100, seed=42)

    # Update the pipelines without assembler and scaler stages
    rf_pipeline = Pipeline(stages=[rf])

    # Train the models
    trained_model = rf_pipeline.fit(train_data)

    evaluator = MulticlassClassificationEvaluator(labelCol="fetal_health", predictionCol="prediction")
    rf_accuracy = evaluator.evaluate(trained_model.transform(test_data), {evaluator.metricName: "accuracy"})

    accuracy_result = rf_accuracy

    print("RandomForest Classifier - Accuracy:", rf_accuracy)

    trained_model.write().overwrite().save("fetal_health_prediction_model")

    # return {"status": "Model trained successfully", "accuracy": rf_accuracy}
    return templates.TemplateResponse("result.html", {"request": request, "status": "Model trained successfully: ✅", "accuracy": rf_accuracy})

@app.post("/predict", response_class=JSONResponse)
async def predict(request: Request,
    baseline_value : float = Form(...),
    accelerations: float = Form(...),
    fetal_movement: float = Form(...),
    uterine_contractions: float = Form(...),
    light_decelerations: float = Form(...),
    severe_decelerations: float = Form(...),
    prolongued_decelerations: float = Form(...),
    abnormal_short_term_variability: float = Form(...),
    mean_value_of_short_term_variability: float = Form(...),
    percentage_of_time_with_abnormal_long_term_variability: float = Form(...),
    mean_value_of_long_term_variability: float = Form(...),
    histogram_width: float = Form(...),
    histogram_min: float = Form(...),
    histogram_max: float = Form(...),
    histogram_number_of_peaks: float = Form(...),
    histogram_number_of_zeroes: float = Form(...),
    histogram_mode: float = Form(...),
    histogram_mean: float = Form(...),
    histogram_median: float = Form(...),
    histogram_variance: float = Form(...),
    histogram_tendency: float = Form(...)
):
    global trained_model
    if not trained_model:
        raise HTTPException(status_code=400, detail="Model not trained. Please train the model first.")
    
    input_data = {
        "baseline_value": baseline_value,
        "accelerations": accelerations,
        "fetal_movement": fetal_movement,
        "uterine_contractions": uterine_contractions,
        "light_decelerations": light_decelerations,
        "severe_decelerations": severe_decelerations,
        "prolongued_decelerations": prolongued_decelerations,
        "abnormal_short_term_variability": abnormal_short_term_variability,
        "mean_value_of_short_term_variability": mean_value_of_short_term_variability,
        "percentage_of_time_with_abnormal_long_term_variability": percentage_of_time_with_abnormal_long_term_variability,
        "mean_value_of_long_term_variability": mean_value_of_long_term_variability,
        "histogram_width": histogram_width,
        "histogram_min": histogram_min,
        "histogram_max": histogram_max,
        "histogram_number_of_peaks": histogram_number_of_peaks,
        "histogram_number_of_zeroes": histogram_number_of_zeroes,
        "histogram_mode": histogram_mode,
        "histogram_mean": histogram_mean,
        "histogram_median": histogram_median,
        "histogram_variance": histogram_variance,
        "histogram_tendency": histogram_tendency
    }

    print("Received form data:", {
        "baseline_value": baseline_value,
        "accelerations": accelerations,
        "fetal_movement": fetal_movement,
        "uterine_contractions": uterine_contractions,
        "light_decelerations": light_decelerations,
        "severe_decelerations": severe_decelerations,
        "prolongued_decelerations": prolongued_decelerations,
        "abnormal_short_term_variability": abnormal_short_term_variability,
        "mean_value_of_short_term_variability": mean_value_of_short_term_variability,
        "percentage_of_time_with_abnormal_long_term_variability": percentage_of_time_with_abnormal_long_term_variability,
        "mean_value_of_long_term_variability": mean_value_of_long_term_variability,
        "histogram_width": histogram_width,
        "histogram_min": histogram_min,
        "histogram_max": histogram_max,
        "histogram_number_of_peaks": histogram_number_of_peaks,
        "histogram_number_of_zeroes": histogram_number_of_zeroes,
        "histogram_mode": histogram_mode,
        "histogram_mean": histogram_mean,
        "histogram_median": histogram_median,
        "histogram_variance": histogram_variance,
        "histogram_tendency": histogram_tendency
    })


    # Preprocess the input data and make predictions
    input_df = pd.DataFrame([input_data])
    input_spark_df = spark.createDataFrame(input_df)

    preprocessed_input = preprocessing_pipeline_model.transform(input_spark_df)

    # Make predictions using the trained model
    predictions = trained_model.transform(preprocessed_input)

    # Collect the predictions
    prediction_result = predictions.collect()

    # Extract the predicted labels
    predicted_labels = [result["prediction"] for result in prediction_result]

    print("Input dataframe:", input_df)
    
    # return {"predictions": predicted_labels[0]}
    if predicted_labels[0]==1:
      predicted_labels = "Normal ✅"
    elif predicted_labels[0]==2:
      predicted_labels = "Suspicious 🤷‍♂️"
    else:
      predicted_labels = "Pathological 💀"
    return templates.TemplateResponse("predict.html", {"request": request, "prediction_result": predicted_labels})


# if __name__ == "__main__":
#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

# ngrok_tunnel = ngrok.connect(8000)
# print('Public URL:', ngrok_tunnel.public_url)
# nest_asyncio.apply()
# uvicorn.run(app, port=8000)


# ngrok_tunnel = ngrok.connect(8000)
# print('New public URL:', ngrok_tunnel.public_url)


ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)
