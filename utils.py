import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample


def plot_sorted_accuracies(data, models, plot=True):
    """
    Plots sorted accuracies of models across features and returns all accuracies in sorted format.
    
    Parameters:
    - data: Dictionary where keys are feature-model combinations and values are accuracy scores.
    - models: List of model names to plot (e.g., ["SVM", "Decision Tree", "Random Forest"])
    
    Returns:
    - List of all accuracies in the format (feature, accuracy, model) sorted by accuracy.
    """
    # Extract features by removing model names
    features = list(set([key.split(" - ")[0] for key in data.keys()]))

    # Initialize accuracy lists for each model
    model_accuracies = {model: [] for model in models}

    # Fill the accuracies for each feature-model pair
    for feature in features:
        for model in models:
            key = f"{feature} - {model} "
            model_accuracies[model].append((feature, data.get(key, 0)))  # (feature, accuracy)

    # Prepare a list to hold all accuracies in the desired format
    all_accuracies = []

    # Collect all accuracies in the specified format
    for model in models:
        for feature, accuracy in model_accuracies[model]:
            all_accuracies.append((feature, accuracy, model))

    # Sort the list of all accuracies by accuracy in descending order
    all_accuracies.sort(key=lambda x: x[1], reverse=True)

    # Prepare data for plotting
    sorted_features = [feature for feature, _ in model_accuracies[models[0]]]  # Sorted by first model
    sorted_accuracies = {model: [acc for _, acc in model_accuracies[model]] for model in models}

    # Bar plot settings
    index = np.arange(len(sorted_features))
    bar_width = 0.2
    if plot :
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot bars for each model
        for i, model in enumerate(models):
            ax.bar(index + i * bar_width, sorted_accuracies[model], bar_width, label=model)

        # Add labels, title, and legend
        ax.set_xlabel('Features')
        ax.set_ylabel('Accuracy')
        ax.set_title('Sorted Model Accuracy Comparison Across Features')
        ax.set_xticks(index + bar_width)
        ax.set_xticklabels(sorted_features, rotation=90)
        ax.legend()

        # Show the plot
        plt.tight_layout()
        plt.show()

    return all_accuracies





def average_accuracy(list1, list2, list3):
    # Combine all lists into a single dictionary to aggregate scores
    accuracy_dict = {}
    
    # Helper function to process each list and update the dictionary
    def process_list(lst):
        for test, accuracy, model in lst:
            if test not in accuracy_dict:
                accuracy_dict[test] = []
            accuracy_dict[test].append(accuracy)

    # Process each input list
    process_list(list1)
    process_list(list2)
    process_list(list3)

    # Calculate average accuracy for each lab test
    average_results = []
    for test, accuracies in accuracy_dict.items():
        avg_accuracy = sum(accuracies) / len(accuracies)  # Renamed to avoid conflict
        average_results.append((test, avg_accuracy))
    
    return average_results




def bootstrap_predict(model, X, y, n_iterations=1000):
    # Store the predictions
    predictions = []
    
    # Perform bootstrapping
    for _ in range(n_iterations):
        # Resample the data
        X_resampled, y_resampled = resample(X, y)
        model.fit(X_resampled, y_resampled)
        
        # Make predictions on the original dataset and ensure numeric type
        preds = model.predict(X)
        preds = np.asarray(preds, dtype=np.float64)  # Ensure numeric array
        predictions.append(preds)
    
    # Convert predictions to a 2D numpy array
    bootstrap_preds = np.array(predictions)
    print("New Shape of bootstrap_preds:", bootstrap_preds.shape)
    print("New Data type of bootstrap_preds:", bootstrap_preds.dtype)
    
    # Calculate mean and confidence intervals
    mean_preds = np.mean(bootstrap_preds, axis=0)
    lower_ci = np.percentile(bootstrap_preds, 2.5, axis=0)
    upper_ci = np.percentile(bootstrap_preds, 97.5, axis=0)

    return mean_preds, lower_ci, upper_ci

