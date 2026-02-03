import mlflow
import dagshub
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

def run_exercise():
    # 1. Connect the Monitor
    dagshub.init(repo_owner='Hundred-Trillion', repo_name='mlops-sentinel', mlflow=True)
    
    with mlflow.start_run(run_name="Spam_Detector_V1"):
        # 2. Simple Dataset (The 'Spam' or 'Ham' list)
        emails = ["Win money now", "Free prize", "Meeting at 5pm", "Hello friend", "Claim your gift"]
        labels = [1, 1, 0, 0, 1] # 1 = Spam, 0 = Ham
        
        # 3. MLOps Logging: Record our data size
        mlflow.log_param("num_samples", len(emails))
        
        # 4. The 'Math': Turn words into numbers
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(emails)
        
        # 5. Training the 'Brain'
        model = MultinomialNB()
        model.fit(X, labels)
        
        # 6. Scoring
        accuracy = model.score(X, labels)
        
        # 7. MLOps Logging: Send the final score to DagsHub
        mlflow.log_metric("accuracy", accuracy)
        
        print(f"✅ Exercise Complete. Accuracy: {accuracy}")

if __name__ == "__main__":
    run_exercise()