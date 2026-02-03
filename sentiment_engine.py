import mlflow
import dagshub
import joblib  # This is the 'Surgical Tool' to save the brain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def build_sentinel():
    dagshub.init(repo_owner='Hundred-Trillion', repo_name='mlops-sentinel', mlflow=True)
    
    with mlflow.start_run(run_name="Sentiment_Sentinel_v1"):
        # 1. The Dataset (Training the Sentinel)
        headlines = [
            "Nvidia stocks soar after AI breakthrough", "Massive layoffs hit tech sector",
            "New battery tech doubles EV range", "Cyber attack leaks millions of records",
            "Startup raises $100M in Series A", "Regulatory crackdown on crypto"
        ]
        labels = ["positive", "negative", "positive", "negative", "positive", "negative"]

        # 2. The Pipeline (The 'Assembly Line')
        # Tfidf turns words into 'importance scores' instead of just counting them
        model = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression())
        ])

        # 3. The Training
        model.fit(headlines, labels)
        
        # 4. Save the 'Brain' locally in the cloud
        joblib.dump(model, 'model_artifact.joblib')
        
        # 5. The Insurance Policy: Log the model file to DagsHub
        mlflow.log_artifact('model_artifact.joblib')
        mlflow.log_metric("headline_count", len(headlines))
        
        print("⚡ Sentinel Trained and Brain Exported to DagsHub!")

if __name__ == "__main__":
    build_sentinel()
    