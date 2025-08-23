import mlflow

def setup_mlflow_run(run_name: str, params: dict, tags: dict):
    # Set the tracking server URI to your local server
    mlflow.set_tracking_uri("http://homeserver:5000")

    # Set the experiment by name. User provided name "Catalanformer" and ID "1".
    experiment_name = "Catalanformer"
    mlflow.set_experiment(experiment_name)

    # Start a new run with the specified name
    active_run = mlflow.start_run(run_name=run_name)
    run_id = active_run.info.run_id
    print(f"✅ MLFlow Run Started:")
    print(f"   Experiment: '{experiment_name}'")
    print(f"   Run Name:   '{run_name}'")
    print(f"   Run ID:     '{run_id}'")

    # Log the initial parameters and set tags
    mlflow.log_params(params)
    mlflow.set_tags(tags)
    print("   Logged initial parameters and tags successfully.")

    return active_run