import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from fermfaultdetect.data import utils
from matplotlib.colors import LinearSegmentedColormap

def set_plot_params(high_res=False):
    '''
    Set the plot parameters for display or saving.
    '''
    if high_res:
        plt.rcParams['figure.dpi'] = 300  # High resolution for saving
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
    else:
        plt.rcParams['figure.dpi'] = 96   # Normal resolution for display
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12

def get_label_dict():
    '''
    Returns a dictionary with the parameter names and their corresponding labels.
    '''
    return {
        'S': 'Substrate [g/L]',
        'X': 'Biomass [g/L]',
        'E': 'Enzyme [g/L]',
        'DO': 'Dissolved Oxygen [Moles O2/L]',
        'DO_saturation': 'Dissolved Oxygen Saturation [%]',
        'F': 'Feed [L/h]',
        'OUR': 'OUR [mol/L/h]',
        'kLa': 'kLa [1/h]',
        'OTR': 'OTR [mol/L/h]',
        'weight': 'Weight [kg]',
        'V': 'Volume [L]',
        'mu': 'Specific Growth Rate [1/h]',
        'mu_app': 'Apparant viscosity [Pa s]',
        'F_set': 'Feed Setpoint [L/h]',
        'air_L': 'Air flow rate [L/h]',
        'air_L_set': 'Air flow rate setpoint [L/h]',
        'N': 'Impeller speed [rpm]',
        'P': 'Agitator power input [W]',
        'defect_steambarrier': 'Defect Steam Barrier',
        'steam_in_feed': 'Feed in Steam',
        'steam_fault': 'Steam Fault',
        'blocked_spargers': 'Blocked Spargers',
        'airflow_OOC': 'Airflow OOC',
        'OUR_OOC': 'OUR OOC',
        'no_fault': 'No Fault',
        'overall': 'Overall',
        'c_f': 'Feed Concentration [g/L]',
    }

def get_thesis_colors():
    '''
    Returns a dictionary with the color codes used in the thesis.
    '''
    thesis_colors = {
    "green": "#048C34",
    "blue": "#005AB5",
    "red": "#DC3220",
    "lemon": "#CE8509",
    "purple": "#A400B3"}

    return thesis_colors

def get_hotcold_colormap():
    '''
    Returns a custom colormap with hot and cold colors.
    '''
    # Define the colors
    colors = ['#005AB5', 'white', '#DC3220']
    # Create the colormap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    return cmap


def plot_single_batch(output):
    """
    Plots the parameters over time from a single DataFrame.

    Parameters:
    - output: pandas DataFrame, containing 't' and tracked variables.
    """
    labels_dict = get_label_dict()
    parameters = output.columns[1:]
    num_rows = (len(parameters) + 1) // 2  # Calculate the number of rows needed
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, num_rows*3), sharex=True)  # Set up a grid of subplots

    # Iterate over parameters and axes
    for i, param in enumerate(parameters):
        row, col = divmod(i, 2)  # Determine the row and column index
        axes[row, col].plot(output['t'], output[param])  # Plot the parameter
        if param in labels_dict:
            axes[row, col].set_ylabel(f"{labels_dict[param]}")  # Set the y-axis label
        axes[row, col].xaxis.set_ticks_position('bottom')
        axes[row, col].tick_params(direction='inout', length=10)
        axes[row, col].ticklabel_format(useOffset=False)
        if param == 'S':
            #Set y limits for S to avoid unwanted scaling
           axes[row, col].set_ylim([0.04, 0.06])

    # If the number of parameters is odd, hide the last subplot
    if len(parameters) % 2 != 0:
        axes[-1, -1].axis('off')  # Hide the last subplot if unused

    if num_rows > 0:  # Check if there are any rows
        axes[-1, 0].set_xlabel('t')  # Set for the bottom left subplot
        if len(parameters) % 2 != 0:
            axes[-2, 1].set_xlabel('t')
        else:
            axes[-1, 1].set_xlabel('t')

    plt.tight_layout()
    plt.subplots_adjust(hspace=.0)
    plt.show()


def plot_multiple_batches(outputs, legends=None, show_legend=False):
    """
    Plots multiple sets of parameters over time from different DataFrames.
    
    Parameters:
    - outputs: List of pandas DataFrames, each containing 't' and other parameters columns.
    - legends: List of strings, labels for each DataFrame for legend.
    """
    # Check if the number of outputs and legends match
    #if len(outputs) != len(legends):
    #    raise ValueError("The number of outputs and legends must match.")

    # Set structure using first df
    labels_dict = get_label_dict()
    parameters = outputs[0].columns[1:]
    num_rows = (len(parameters) + 1) // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, num_rows*3), sharex=True)
    
    # Colors or styles for differentiating lines, adjust based on preference or number of outputs
    colors = plt.cm.tab10(np.linspace(0, 1, len(outputs)))
    
    # Iterate over parameters and axes
    for i, param in enumerate(parameters):
        row, col = divmod(i, 2)
        for j, output in enumerate(outputs):
            # Plot parameter for each output on the same axes
            axes[row, col].plot(output['t'], output[param], label=f"{legends[j]} - {param}", color=colors[j])
        if param in labels_dict:
            axes[row, col].set_ylabel(f"{labels_dict[param]}")
        if param == 'S':
            #Set y limits for S to avoid unwanted scaling
           axes[row, col].set_ylim([0.04, 0.06])
        #axes[row, col].ticklabel_format(style='plain', axis='both')
        axes[row, col].ticklabel_format(useOffset=False)
        if show_legend is True:
            axes[row, col].legend()
        axes[row, col].xaxis.set_ticks_position('bottom')
        axes[row, col].tick_params(direction='inout', length=10)
    
    # If the number of parameters is odd, hide the last subplot
    if len(parameters) % 2 != 0:
        axes[-1, -1].axis('off')
    
    if num_rows > 0:
        axes[-1, 0].set_xlabel('t')
        if len(parameters) % 2 != 0:
            axes[-2, 1].set_xlabel('t')
        else:
            axes[-1, 1].set_xlabel('t')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=.0)
    plt.show()

def plot_batchset_statistics(stats_df, parameters, show_std=True, show_minmax=True, separate_plots=False, color="tab:blue", dpi=96):
    """
    Plots the the mean, standard deviation and min/max 

    Parameters:
    - batch_list: List of pandas DataFrames, each containing 't' and other parameters columns.
    """
    labels_dict = get_label_dict()
    #set_plot_params()
    # Set structure using first df
    #parameters = batch_list[0].columns[1:]
    if separate_plots==False:
        num_rows = (len(parameters) + 1) // 2
        fig, axes = plt.subplots(num_rows, 2, figsize=(10, num_rows*3), sharex=True)
        fig.dpi = dpi
    
    # Iterate over parameters and axes
    for i, param in enumerate(parameters):

        if separate_plots:
            fig, axes = plt.subplots(1, 1, figsize=(8, 6))
            fig.dpi = dpi
        else:
            row, col = divmod(i, 2)
            axes = axes[row, col]     
        means = stats_df[(param, 'mean')]
        if show_std:
            stds = stats_df[(param, 'std')]
        if show_minmax:
            mins = stats_df[(param, 'min')]
            maxs = stats_df[(param, 'max')]
        axes.plot(stats_df[('t', 'mean')], means, label=f"mean({param})", color = color)
        if show_std:
            axes.fill_between(stats_df[('t', 'mean')], means - stds, means + stds, alpha=0.3, label=f"std({param})", color = color)
        if show_minmax:
            axes.plot(stats_df[('t', 'mean')], mins, linestyle='--', linewidth = 0.8, alpha = 1, color = color, label=f"min({param})")
            axes.plot(stats_df[('t', 'mean')], maxs, linestyle='--', linewidth = 0.8, alpha = 1, color = color, label=f"max({param})")
        if param in labels_dict:
            axes.set_ylabel(f"{labels_dict[param]}")
        else:
            axes.set_ylabel(f"{param}")
        axes.set_xlabel('Time [h]')
        axes.xaxis.set_ticks_position('bottom')
        axes.tick_params(direction='inout', length=10)
        axes.ticklabel_format(useOffset=False)
        axes.set_xlim(left=0)
        axes.set_ylim(bottom=0)

    # If the number of parameters is odd, hide the last subplot
    if separate_plots==False:
        if len(parameters) % 2 != 0:
            axes[-1, -1].axis('off')
        
        if num_rows > 0:
            axes[-1, 0].set_xlabel('t')
            if len(parameters) % 2 != 0:
                axes[-2, 1].set_xlabel('t')
            else:
                axes[-1, 1].set_xlabel('t')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=.0)
    plt.show()


def plot_input_distributions(df):
    """
    Plot a histogram for each variable in the dataframe.

    :param df: pandas DataFrame with each column representing a variable.
    """
    for column in df.columns:
        # Plot histogram
        plt.figure(figsize=(6, 4))
        plt.hist(df[column], bins=20, alpha=0.7)  # Bins can be adjusted
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

def plot_parameter_distribution(batch_list, parameter, time_value, bins = 10):
    """
    Plots a histogram of the values for a given parameter at the time closest to the specified time_value
    across multiple DataFrames, where the time is stored in a column named 't'.
    
    Args:
    dataframes (list of pd.DataFrame): List of DataFrames with a time column 't' and parameters as columns.
    parameter (str): The name of the parameter to plot.
    time_value (int or float): The time value to find the closest match in each DataFrame.
    
    """
    values = []
    
    for df in batch_list:
        # Check if the parameter exists in the DataFrame
        if parameter in df.columns:
            # Find the index of the row with the closest 't' value to the specified time value
            closest_idx = (df['t'] - time_value).abs().idxmin()
            # Extract the value for the parameter at the closest time
            value = df.at[closest_idx, parameter]
            values.append(value)
        else:
            print(f"Parameter {parameter} not found in one of the DataFrames.")
    
    # Plotting the histogram of the values
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=10, alpha=0.7) # color='blue', edgecolor='black'
    plt.title(f'Histogram of {parameter} at {time_value} h')
    plt.xlabel(f'Value of {parameter}')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def plot_batchfolder(set_dir):
    '''
    Plots all batches of a set inside a folder

    Parameters:
    - set_dir: Directory containing CSV files for each batch.
    '''
    batch_list, legend_list = utils.load_batchfolder(set_dir)

    # Plot the results
    plot_multiple_batches(batch_list, legend_list, show_legend=False)