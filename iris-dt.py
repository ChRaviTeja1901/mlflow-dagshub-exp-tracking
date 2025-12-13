import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import os
import shutil

def main():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    max_depth = 20

    
    dagshub.init(repo_owner='ChRaviTeja1901', repo_name='mlflow-dagshub-exp-tracking', mlflow=True)


    mlflow.set_tracking_uri("https://dagshub.com/ChRaviTeja1901/mlflow-dagshub-exp-tracking.mlflow")

    # Set the MLflow experiment name, every run will be recorded under this experiment and if the experiment does not exist it will be created
    mlflow.set_experiment("Iris Decision Tree Classifier")

    # one more option to set experiment is to set experiment id directly with in context manager
    # with mlflow.start_run(experiment_id="1"):

    # Default run name will be assigned automatically but you can also set a custom run name
    # with mlflow.start_run(run_name="My Decision Tree Run"):

    with mlflow.start_run():
        dt = DecisionTreeClassifier(max_depth=max_depth)
        model = dt.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("test_size", test_size)

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_artifact(__file__)

        # Save the sklearn model locally and log the model directory as artifacts.
        # Dagshub's MLflow endpoint does not support the model-registry REST endpoint
        # MLflow's `log_model` may call that endpoint; to avoid the unsupported endpoint,
        # save the model and then upload the model folder as artifacts.
        model_dir = "decision_tree_model"
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        mlflow.sklearn.save_model(model, model_dir)
        mlflow.log_artifacts(model_dir, artifact_path="decision_tree_model")
        shutil.rmtree(model_dir)
        mlflow.set_tag("model", "Decision Tree")
        # print(mlflow.get_artifact_uri())



if __name__ == "__main__":
    main()