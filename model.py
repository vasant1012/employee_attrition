from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class ModelConfig:
    def __init__(self, model_type='random_forest', random_state=42, encoder=None, scaler=None):
        """
        Initialize a ModelConfig object.

        Parameters:
        model_type (str): The type of model to be used. Default is 'random_forest'.
                        Supported types are 'random_forest', 'xgboost', and 'logistic_regression'.
        random_state (int): The random seed for reproducibility. Default is 42.

        Returns:
        None
        """
        self.model_type = model_type
        self.random_state = random_state
        if encoder is None:
            self.encoder = OneHotEncoder()
        if scaler is None:
            self.scaler = StandardScaler()

    def get_model(self):
        """
        Returns the initialized model based on the model type specified in the configuration.

        Parameters:
        self (ModelConfig): The ModelConfig object containing the model type and random state.

        Returns:
        model (RandomForestClassifier | XGBClassifier | LogisticRegression): The initialized model.

        Raises:
        ValueError: If an unsupported model type is specified in the configuration.
        """
        if self.model_type == 'random_forest':
            return RandomForestClassifier(random_state=self.random_state)
        elif self.model_type == 'xgboost':
            return XGBClassifier(use_label_encoder=False,
                                 eval_metric='logloss',
                                 random_state=self.random_state)
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
    def create_ml_pipeline(df, target_column, encoder=None, scaler=None, model=None):
        """
        Create a preprocessing pipeline for a dataframe.
        
        Parameters:
        - df: pd.DataFrame - The input dataframe.
        - target_column: str - The name of the target column.
        - encoder: object - The encoder to use for categorical features (default is OneHotEncoder).
        - scaler: object - The scaler to use for numerical features (default is StandardScaler).
        - model: object - The model to use for classification (default is RandomForestClassifier).
        
        Returns:
        - pipeline: Pipeline - The preprocessing and modeling pipeline.
        - X_train, X_test, y_train, y_test: pd.DataFrame - Split training and testing data.
        """
        # Default encoder and scaler
        if encoder is None:
            encoder = OneHotEncoder()
        if scaler is None:
            scaler = StandardScaler()
        if model is None:
            model = RandomForestClassifier()
        
        # Identify categorical and numerical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        # Remove the target column from feature columns
        if target_column in categorical_columns:
            categorical_columns.remove(target_column)
        if target_column in numerical_columns:
            numerical_columns.remove(target_column)
        
        # Create the column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', scaler, numerical_columns),
                ('cat', encoder, categorical_columns)
            ]
        )
        
        # Create the pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Split the data into training and test sets
        X = df.drop(columns=[target_column], axis=1)
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return pipeline, X_train, X_test, y_train, y_test
