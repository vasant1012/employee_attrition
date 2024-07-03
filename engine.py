import pandas as pd
from logger import logger
from train import train, evaluate
from model import ModelConfig
import warnings
warnings.filterwarnings('ignore')

# Train and evaluate the model
config = ModelConfig(model_type='xgboost')
model = config.get_model()
df = pd.read_csv('train.csv')
pipeline, X_train, X_test, y_train, y_test = config.create_ml_pipeline(df,
                                                                'Attrition')

# Train the pipeline
logger.info('Model training started!')
model = train(pipeline, X_train, y_train)
logger.info('Model training is completed!')

# Make predictions
accuracy, report = evaluate(model, X_test, y_test)
logger.info(f'Accuracy: {round(accuracy * 100, 2)}%')
print('Report:', report)