import pandas as pd
from train import train, evaluate
from model import ModelConfig
from engine import create_ml_pipeline

# Train and evaluate the model
config = ModelConfig()
model = config.get_model()
df = pd.read_csv('train.csv')
pipeline, X_train, X_test, y_train, y_test = create_ml_pipeline(df, 'Attrition', encoder=None, scaler=None, model=None)

# Train the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
predictions = pipeline.predict(X_test)