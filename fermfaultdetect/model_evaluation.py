import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from fermfaultdetect.utils import get_simulation_dir, get_root
from fermfaultdetect.data.utils import load_batchset, dataloader
import json
from itertools import zip_longest
from fermfaultdetect.visualizations import visualize

def detection_accuracy(FDR, FAR):
    '''
    Returns the accuracy of the model based on the False Detection Rate (FDR) and False Alarm Rate (FAR).
    '''
    accuracy = (FDR+(1-FAR))/(FDR+(1-FDR)+FAR+(1-FAR))
    return accuracy

def metrics_table_oneclass(test, pred, save_path=None):
    '''
    Returns FDR, FAR and accuracy for each fault type and overall for models trained for fault detection.

    Parameters:
    test (pd.DataFrame): test array with true fault for each individual fault type.
    pred (np.array or DataFrame): array with predicted values as a one class problem.
    save_path (String): If not None, metrics will be saved here as a csv.

    Returns:
    ratios_df (pd.DataFrame): DataFrame with FDR, FAR and accuracy for each fault type and overall.
    '''
    # Check if pred is a DataFrame, if yes, convert to np.array
    if isinstance(pred, pd.DataFrame):
        if 'no_fault' in pred.columns: # Delete "no_fault" column
            pred = pred.drop(columns='no_fault')
        pred = pred.to_numpy()
        
    # Delete "no_fault" column
    if 'no_fault' in test.columns:
        test = test.drop(columns='no_fault')
    test_one_class = test.max(axis=1).astype(int)
    cm_one_class = confusion_matrix((1 - test_one_class), (1 - pred), normalize='true')
    FDR_one_class = cm_one_class[0, 0]
    FAR_one_class = cm_one_class[1, 0]
    acc_one_class = detection_accuracy(FDR_one_class, FAR_one_class)

    ratios = {}

    for column in test.columns:
        tp = np.sum((test[column] == 1) & (pred == 1))
        #fp = np.sum((test[column] == 1) & (pred == 0))
        
        total_actual_positives = np.sum(test[column] == 1)
        #total_actual_negatives = np.sum(test[column] == 0)
        
        tp_ratio = tp / total_actual_positives if total_actual_positives else 0
        #fp_ratio = fp / total_actual_negatives if total_actual_negatives else 0
        #acc = detection_accuracy(tp_ratio, fp_ratio)
        
        #ratios[column] = {'Fault detection rate': tp_ratio, 'False alarm rate': fp_ratio, 'Accuracy': acc}
        ratios[column] = {'Fault detection rate': tp_ratio}

    # Add overall metrics
    ratios['overall'] = {'Fault detection rate': FDR_one_class, 'False alarm rate': FAR_one_class, 'Accuracy': acc_one_class}

    if save_path:        
        pd.DataFrame(ratios).T.to_csv(save_path)

    ratios_df = pd.DataFrame(ratios).T
    
    return ratios_df


def metrics_table_multclass(test, pred, show_confusion_matrix = False, save_path=None):
    '''
    Returns FDR, FAR and accuracy for each fault type and overall for models trained for fault diagnosis.

    Parameters:
    test (pd.DataFrame): test array with true fault for each individual fault type.
    pred (np.array or DataFrame): array with predicted values as a one class problem.
    save_path (String): If not None, metrics will be saved here as a csv.

    Returns:
    ratios_df (pd.DataFrame): DataFrame with FDR, FAR and accuracy for each fault type and overall.
    '''

    pred = pred[test.columns]
    columns = test.columns

    cm = confusion_matrix(test.values.argmax(axis=1), pred.values.argmax(axis=1), normalize="true")

    metrics = {}
    no_fault_idx = columns.get_loc("no_fault")
    fault_columns = columns.drop("no_fault")
    for fault_column in fault_columns:
        idx = columns.get_loc(fault_column)
        FDR = cm[idx, idx]
        FAR = cm[no_fault_idx, idx]
        acc = detection_accuracy(FDR, FAR)
        metrics[fault_column] = {"Fault detection rate": FDR, "False alarm rate": FAR, "Accuracy": acc}
        
    test_all = test.drop(columns="no_fault").max(axis=1).astype(int)
    pred_all = pred.drop(columns="no_fault").max(axis=1).astype(int)
    cm_all = confusion_matrix((1 - test_all), (1 - pred_all), normalize='true')
    FDR_all = cm_all[0, 0]
    FAR_all = cm_all[1, 0]
    acc_all = detection_accuracy(FDR_all, FAR_all)
    metrics["overall"] = {"Fault detection rate": FDR_all, "False alarm rate": FAR_all, "Accuracy": acc_all}

    metrics_df = pd.DataFrame(metrics).T

    if save_path:        
        pd.DataFrame(metrics_df).T.to_csv(save_path)

    labels_dict = visualize.get_label_dict()

    labels = [labels_dict.get(col, col) for col in columns]

    visualize.set_plot_params(high_res=False)

    if show_confusion_matrix:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot() 
        plt.xticks(rotation=45, ha='right')
        fig = disp.figure_
        fig.set_figwidth(8)
        fig.set_figheight(8)
        fig.set_dpi(300)
        #plt.tight_layout()
        #plt.figure(figsize=(10, 10))
        #plt.rcParams['figure.dpi'] = 300
        #if save_path:
        #    plt.savefig(os.path.join(save_path, "confusion_matrix.png"), dpi=300)
        plt.show()
    
    return metrics_df


def visualize_metrics(ratios_df, save_path=None):
    '''
    Visualizes the fault detection rates and overall metrics in a bar plot and heatmap, respectively.

    Parameters:
    ratios_df (pd.DataFrame): DataFrame with FDR, FAR and accuracy for each fault type and overall.
    '''
    # Set normal plot parameters for display
    visualize.set_plot_params(high_res=False)
    colors = visualize.get_thesis_colors()

    # Separate individual fault detection rates and overall metrics
    fault_detection_df = ratios_df.drop('overall')
    overall_metrics_df = ratios_df.loc[['overall']]

    # Plot fault detection rates
    plt.figure(figsize=(8, 6))
    sns.barplot(x=fault_detection_df.index, y=fault_detection_df['Fault detection rate'], color = colors["green"])
    plt.title('Fault Detection Rate for Each Fault Type')
    plt.xlabel('Fault Type')
    plt.ylabel('Fault Detection Rate')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Create colormap
    from  matplotlib.colors import LinearSegmentedColormap
    c = ["darkred","red","lightcoral","white", "palegreen","green","darkgreen"]
    v = [0,.15,.4,.5,0.6,.9,1.]
    l = list(zip(v,c))
    cmap=LinearSegmentedColormap.from_list('rg',l, N=256)

    # Plot overall metrics as a heatmap
    plt.figure(figsize=(5, 3))
    sns.heatmap(overall_metrics_df, vmin=0, vmax=1, annot=True, cmap=cmap, cbar=False)
    plt.title('Overall Metrics')
    plt.tight_layout()
    plt.show()

    # Set high resolution parameters for saving if a path is provided
    if save_path:
        visualize.set_plot_params(high_res=True)  # Switch to high resolution settings
        plt.figure(figsize=(8, 6))
        sns.barplot(x=fault_detection_df.index, y=fault_detection_df['Fault detection rate'], color = colors["green"])
        #plt.title('Fault Detection Rate for Each Fault Type')
        #plt.xlabel('Fault Type')
        plt.ylabel('Fault Detection Rate')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "fault_detection_rates.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(5, 3))
        sns.heatmap(overall_metrics_df, vmin=0, vmax=1, annot=True, cmap='coolwarm', cbar=False)
        plt.title('Overall Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "overall_metrics.png"), dpi=96)
        plt.close()



def plot_example_set(model, train_dl, setname, parameter_plotted, combined_figure=True, feature_engineering=False):
    '''
    Visualizes fault event predictions of a model trained for fault detection on an exemplatory set.

    Parameters:
    model: trained model
    train_dl: dataloader used for training the model
    setname: name of the exemplatory set
    parameter_plotted: parameter to be plotted
    combined_figure: if True, all plots are combined into one figure
    feature_engineering: if True, the airflow preprocessing is applied to the data
    '''
    sim_dir = get_simulation_dir()
    data_path = os.path.join(sim_dir, setname)
    data_set = load_batchset(data_path)

    target_cols = ['defect_steambarrier', 'steam_in_feed', 'blocked_spargers', 'airflow_OOC', 'OUR_OOC', 'no_fault'] # set target columns
    data_dl = dataloader(batchset=data_set)
    if feature_engineering:
        data_dl.apply_airflow_preprocess(target_cols)
    data_dl.import_standardization(train_dl)
    data_dl.standardize_data(exclude_cols=target_cols)
    data_X, data_Y = data_dl.get_data(split_batches=False, target_cols=target_cols, separate_target_matrix=True, fuse_target_cols=True)

    predictions = model.predict(data_X)

    # Concatenate predictions and data_X sideways
    data_X = data_dl.invert_standardization(data_X)
    pred_data = pd.concat([data_X, predictions], axis=1)

    pred_data = data_dl.split_into_batches(pred_data)

    monte_carlo_dir = os.path.join(get_root(), 'data/Monte_Carlo')
    faultconfig_path = os.path.join(monte_carlo_dir, setname + '_faultconfig.json')
    bioconfig_path = os.path.join(monte_carlo_dir, setname + '_bioconfig.csv')

    with open(faultconfig_path, 'r') as file:
        faultconfig = json.load(file)

    bioconfig = pd.read_csv(bioconfig_path)

    # Plotting
    if combined_figure:
        num_plots = len(pred_data)
        fig, axes = plt.subplots(num_plots, 1, figsize=(8, 5 * num_plots), sharex=True)

    fault_colors = {
        "defect_steambarrier": "red",
        "steam_in_feed": "brown",
        "blocked_spargers": "green",
        "airflow_OOC": "orange",
        "OUR_OOC": "purple"
    }
    label_dict = visualize.get_label_dict()

    for i, (fault, (_, bio), batch) in enumerate(zip_longest(faultconfig, bioconfig.iterrows(), pred_data, fillvalue=None)):
        if combined_figure:
            ax = axes[i]
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        fault_type = fault['event_type'] if fault is not None else None
        fault_color = fault_colors.get(fault_type, 'black') if fault_type is not None else 'black'

        if parameter_plotted not in batch.columns:
            print(f"Parameter {parameter_plotted} not found in batch columns.")
            continue

        t = batch['t']
        param_values = batch[parameter_plotted]

        ax.plot(t, param_values, label=parameter_plotted)

        if fault is not None:
            # Add fault event start and end lines
            if 't_start' in fault and 'duration' in fault:
                t_start = fault['t_start']
                t_end = t_start + fault['duration']
            else:
                t_start = 0
                t_end = None

            # Special case for blocked spargers, since the fault persists after t_end
            if fault_type == "blocked_spargers":
                t_end = None

            ax.axvline(x=t_start, color=fault_color, linestyle='--', label=f"{fault_type} start")
            if t_end is not None and t_end < t.max():
                ax.axvline(x=t_end, color=fault_color, linestyle='--', label=f"{fault_type} end")

        # Highlight the area during detected faults in batch data
        if "fault" in batch.columns:
            fault_detected = batch["fault"].astype(bool)
            ax.fill_between(t, param_values.min(), param_values.max(), where=fault_detected, color="royalblue", alpha=0.3)

        # Set title and labels
        ax.set_title(f"Fault Type: {label_dict[fault_type]}" if fault_type else "No Fault")
        ax.set_ylabel(label_dict[parameter_plotted])
        #ax.legend()

        if not combined_figure:
            ax.set_xlabel("Time [h]")
            plt.tight_layout()
            plt.show()

            # Print the summary of the configuration below each plot
        #config_summary = json.dumps(fault, indent=4)
        #ax.text(0.5, -0.15, config_summary, ha='center', va='top', transform=ax.transAxes, fontsize=8)

    if combined_figure:
        axes[-1].set_xlabel("Time")
        plt.tight_layout()
        plt.show()

def plot_example_set_multclass(model, dataset_name, parameter_plotted, target_cols, combine_steam_faults=False, combined_figure=True, show_legend=True):
    '''
    Visualizes fault event predictions of a model trained for fault diagnosis on an exemplatory set.

    Parameters:
    model: trained model
    dataset_name: name of the exemplatory set
    parameter_plotted: parameter to be plotted
    target_cols: target columns
    combine_steam_faults: if True, steam faults are combined into one fault type
    combined_figure: if True, all plots are combined into one figure
    show_legend: if True, a legend is shown in the plots
    '''
    sim_dir = get_simulation_dir()
    data_path = os.path.join(sim_dir, dataset_name)
    data_set = load_batchset(data_path)

    #target_cols = ['defect_steambarrier', 'steam_in_feed', 'blocked_spargers', 'airflow_OOC', 'OUR_OOC', 'no_fault'] # set target columns
    data_dl = dataloader(batchset=data_set)
    if combine_steam_faults:
        data_dl.combine_fault_events(['defect_steambarrier', 'steam_in_feed'], 'steam_fault')
    data_dl.apply_airflow_preprocess(target_cols)
    data_dl.standardize_data(exclude_cols=target_cols)
    data_X, data_Y = data_dl.get_data(split_batches=False, target_cols=target_cols, separate_target_matrix=True, fuse_target_cols=False)

    predictions = model.predict(data_X)

    # Concatenate predictions and data_X sideways
    data_X = data_dl.invert_standardization(data_X)
    pred_data = pd.concat([data_X, predictions], axis=1)

    # Debug: Check data before splitting
    # print("Concatenated Data (before splitting):")
    # print(pred_data.head())

    pred_data = data_dl.split_into_batches(pred_data)

    # Debug: Check each batch to ensure it's not empty
    # for i, batch in enumerate(pred_data):
    #     print(f"Batch {i+1}:")
    #     print(batch.head())

    monte_carlo_dir = os.path.join(get_root(), 'data/Monte_Carlo')
    faultconfig_path = os.path.join(monte_carlo_dir, dataset_name + '_faultconfig.json')
    bioconfig_path = os.path.join(monte_carlo_dir, dataset_name + '_bioconfig.csv')

    with open(faultconfig_path, 'r') as file:
        faultconfig = json.load(file)

    if combine_steam_faults:
        for fault in faultconfig:
            if fault['event_type'] == 'defect_steambarrier':
                fault['event_type'] = 'steam_fault'
            elif fault['event_type'] == 'steam_in_feed':
                fault['event_type'] = 'steam_fault'

    bioconfig = pd.read_csv(bioconfig_path)

    # Plotting
    if combined_figure:
        num_plots = len(pred_data)
        fig, axes = plt.subplots(num_plots, 1, figsize=(8, 4 * num_plots), sharex=True)
    if combine_steam_faults:
        fault_colors = {
            "steam_fault": "red",
            "blocked_spargers": "green",
            "airflow_OOC": "orange",
            "OUR_OOC": "purple"
        }
    else:
        fault_colors = {
            "defect_steambarrier": "red",
            "steam_in_feed": "brown",
            "blocked_spargers": "green",
            "airflow_OOC": "orange",
            "OUR_OOC": "purple"
        }

    fault_events = target_cols
    fault_events.remove('no_fault')

    for i, (fault, (_, bio), batch) in enumerate(zip_longest(faultconfig, bioconfig.iterrows(), pred_data, fillvalue=None)):
        if combined_figure:
            ax = axes[i]
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        fault_type = fault['event_type'] if fault is not None else None
        fault_color = fault_colors.get(fault_type, 'black') if fault_type is not None else 'black'

        if parameter_plotted not in batch.columns:
            print(f"Parameter {parameter_plotted} not found in batch columns.")
            continue

        t = batch['t']
        param_values = batch[parameter_plotted]

        ax.plot(t, param_values, label=parameter_plotted)

        if fault is not None:
            # Add fault event start and end lines
            if 't_start' in fault and 'duration' in fault:
                t_start = fault['t_start']
                t_end = t_start + fault['duration']
            else:
                t_start = 0
                t_end = None

            # Special case for blocked spargers, since the fault persists after t_end
            if fault_type == "blocked_spargers":
                t_end = None

            ax.axvline(x=t_start, color=fault_color, linestyle='--', label=f"{fault_type} start")
            if t_end is not None and t_end < t.max():
                ax.axvline(x=t_end, color=fault_color, linestyle='--', label=f"{fault_type} end")

        # Highlight the area during detected faults in batch data
        for fault_event in fault_events:
            if fault_event in batch.columns:
                fault_detected = batch[fault_event].astype(bool)
                if fault_detected.any():
                    ax.fill_between(t, param_values.min(), param_values.max(), where=fault_detected, color=fault_colors[fault_event], alpha=0.3, label=fault_event)
        # if fault_type in batch.columns:
        #     fault_detected = batch[fault_type].astype(bool)
        #     ax.fill_between(t, param_values.min(), param_values.max(), where=fault_detected, color=fault_color, alpha=0.3)

        # Set title and labels
        ax.set_title(f"Fault Type: {fault_type}" if fault_type else "No Fault")
        ax.set_ylabel(parameter_plotted)
        if show_legend:
            ax.legend()

        if not combined_figure:
            ax.set_xlabel("Time")
            plt.tight_layout()
            plt.show()

            # Print the summary of the configuration below each plot
        #config_summary = json.dumps(fault, indent=4)
        #ax.text(0.5, -0.15, config_summary, ha='center', va='top', transform=ax.transAxes, fontsize=8)
    if combined_figure:
        axes[-1].set_xlabel("Time")
        plt.tight_layout()
        plt.show()