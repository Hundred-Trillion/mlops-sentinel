import mlflow
import dagshub
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

def train_model():
    # Connect to DagsHub
    dagshub.init(repo_owner='Hundred-Trillion', repo_name='mlops-sentinel', mlflow=True)
    
    with mlflow.start_run(run_name="Codespaces_to_Colab_Run"):
        # 1. Load Data
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # 2. Set "Settings" (Hyperparameters)
        n_estimators = 100
        mlflow.log_param("n_estimators", n_estimators)
        
        # 3. Train
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X, y)
        
        # 4. Log Result
        accuracy = clf.score(X, y)
        mlflow.log_metric("accuracy", accuracy)
        
        print(f"Success! Accuracy: {accuracy}")

if __name__ == "__main__":
    train_model()