import json
from datetime import datetime

def get_user_input():
    experiment_info = {}
    
    # Experiment ID and date
    print("Enter Experiment ID: 1")
    experiment_info["experiment_id"] = 1#input("Enter Experiment ID: ")
    experiment_info["date"] = str(datetime.now())
    
    # Researcher information
    print("Enter Researcher Name: wwjang")
    experiment_info["researcher"] = "wwjang"#input("Enter Researcher Name: ")
    print("Enter Experiment Goal: FL")
    experiment_info["experiment_goal"] = 'FL'#input("Enter Experiment Goal: ")

    # System configuration
    experiment_info["system"] = {
        "server_configuration": input("Enter Server Configuration: "),
        "client_configuration": input("Enter Client Configuration: "),
        "operating_system": input("Enter Operating System: "),
        "python_version": input("Enter Python Version: "),
        "packages": input("Enter Python Packages and Versions (comma-separated): ")
    }

    # Dataset information
    experiment_info["dataset"] = {
        "dataset_name": input("Enter Dataset Name: "),
        "dataset_size": input("Enter Dataset Size: "),
        "data_distribution": input("Enter Data Distribution (e.g., IID, non-IID): ")
    }

    # Model configuration
    experiment_info["model"] = {
        "model_architecture": input("Enter Model Architecture: "),
        "model_parameters": input("Enter Number of Model Parameters: "),
        "pretrained": input("Is the model pretrained? (Yes/No): ")
    }

    # Training configuration
    experiment_info["training"] = {
        "epochs_per_round": input("Enter Number of Epochs per Round: "),
        "batch_size": input("Enter Batch Size: "),
        "learning_rate": input("Enter Learning Rate: "),
        "optimizer": input("Enter Optimizer: "),
        "loss_function": input("Enter Loss Function: "),
        "rounds": input("Enter Number of Rounds: "),
        "aggregation_method": input("Enter Aggregation Method: ")
    }

    # Communication configuration
    experiment_info["communication"] = {
        "communication_rounds": input("Enter Number of Communication Rounds: "),
        "clients_per_round": input("Enter Number of Clients per Round: "),
        "network_configuration": input("Enter Network Configuration: "),
        "bandwidth": input("Enter Bandwidth: ")
    }

    # Results (these can be updated after the experiment)
    experiment_info["results"] = {
        "initial_accuracy": input("Enter Initial Accuracy: "),
        "final_accuracy": input("Enter Final Accuracy: "),
        "loss": input("Enter Loss: "),
        "training_time": input("Enter Training Time: "),
        "communication_time": input("Enter Communication Time: "),
        "other_metrics": input("Enter Other Metrics (comma-separated): ")
    }

    # Additional notes
    experiment_info["notes"] = {
        "challenges": input("Enter Challenges: "),
        "observations": input("Enter Observations: "),
        "future_work": input("Enter Suggestions for Future Work: ")
    }

    return experiment_info

def save_experiment_info(experiment_info, filename):
    with open(filename, 'w') as f:
        json.dump(experiment_info, f, indent=4)
    print(f"Experiment information saved to {filename}")

def main():
    experiment_info = get_user_input()
    filename = f"experiment_{experiment_info['experiment_id']}.json"
    save_experiment_info(experiment_info, filename)

if __name__ == "__main__":
    main()
