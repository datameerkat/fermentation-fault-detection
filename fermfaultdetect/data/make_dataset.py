import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import os
from fermfaultdetect.process_models import fed_batch_model
from fermfaultdetect.data import preprocessing
from fermfaultdetect.utils import get_root, get_simulation_dir
import json
import ipywidgets as widgets
from IPython.display import display


def monte_carlo_inputs(input_distribution, num_sets, file_dir, set_name):
    """
    Generate multiple sets of input variables based on their distributions and store in a DataFrame.

    :param input_distribution: Dict with variable names as keys and tuples (median, range, distribution form) as values.
    :param num_sets: Number of sets to generate.
    :return: DataFrame where each row represents a set of input variables.
    """
    sets_list = [] 
    for _ in range(num_sets):
        set_dict = {}
        for var_name, (median, range_val, dist_form) in input_distribution.items():
            if dist_form == 'normal':
                distribution = np.random.normal(loc=median, scale=range_val, size=1)
            elif dist_form == 'uniform':
                lower, upper = median - range_val, median + range_val
                distribution = np.random.uniform(low=lower, high=upper, size=1)
            elif dist_form == 'exponential':
                rate = 1.0 / median  # Transform median to rate parameter
                distribution = np.random.exponential(scale=1.0/rate, size=1)
            elif dist_form == 'truncated_normal':
                lower, upper = median - range_val, median + range_val
                sd = (upper - lower) / 4  # Standard deviation assuming 95 % of the data are within the lower and upper bound
                a, b = (lower - median) / sd, (upper - median) / sd
                distribution = truncnorm(a, b, loc=median, scale=sd).rvs(size=1)
            elif dist_form == 'constant':
                distribution = [median]
            else:
                raise ValueError(f"Unknown distribution form '{dist_form}'")
            # Add more distribution forms here as needed
            set_dict[var_name] = distribution[0]
        sets_list.append(set_dict)  # Add the generated set to the list
    
    # Convert the list of dictionaries into a DataFrame
    sets_df = pd.DataFrame(sets_list)
    # Save dataframe as csv file
    monte_carlo_filename = (set_name + '_monte_carlo.csv')
    monte_carlo_location = os.path.join(file_dir, monte_carlo_filename)
    sets_df.to_csv(monte_carlo_location, index=False)
    return sets_df


def generate_dataset(init_config, filepath_monte_carlo, set_name, output_dir=get_simulation_dir()):
    '''
    Perform Monte Carlo Simulation for one recipe based on set of biokinetic parameters.

    Parameters:
    - input_config: Dictionary with controlable parameters.
    - monte_carlo_location: Path to the CSV file with input parameters.
    - set_name: Name of the set of batches.
    - output_dir: Directory to save the output CSV files.
    '''

    # Generate output folder based on your set name
    output_folder = os.path.join(output_dir, set_name)
    os.makedirs(output_dir, exist_ok=True)
    # Read set of inputs from Monte Carlo simulation
    input = pd.read_csv(filepath_monte_carlo)

    for index, row in input.iterrows():
        # Create the configuration for this batch
        bio_config = {
            'X0': row['X0'],
            'C': row['C'],
            'Y_SX': row['Y_SX'],
            'Y_SE': row['Y_SE'],
            'Y_SO': row['Y_SO'],
            'Y_SC': row['Y_SC'],
            'Y_XS_true': row['Y_XS_true'],
            'Y_XO_true': row['Y_XO_true'],
            'm_s': row['m_s'],
            'm_o': row['m_o'],
        }
        
        # Initialize and run the model
        model = fed_batch_model.A_oryzae(init_config, bio_config)
        output = model.control_solve()
        
        # Save the output to a CSV
        output_file_path = os.path.join(output_folder, f'batch_{index+1}.csv')
        output.to_csv(output_file_path, index=False)

def monte_carlo_biokinetic(input_distribution, num_sets, file_dir, set_name, seed=42):
    """
    Generate multiple sets of input variables based on their distributions and store in a DataFrame.

    :param input_distribution: Dict with variable names as keys and tuples (median, uncertainty (in %), distribution form) as values.
    :param num_sets: Number of sets to generate.
    :return: DataFrame where each row represents a set of input variables.
    """
    np.random.seed(seed)
    sets_list = []
    for i in range(num_sets):
        set_dict = {}
        for j, (var_name, (median, range_percent, dist_form)) in enumerate(input_distribution.items()):
            local_seed = seed + i * len(input_distribution) + j # robust seeding
            min, max = median - median * range_percent / 100, median + median * range_percent / 100
            distribution = sample_distribution(min, max, dist_form, seed=local_seed)
            set_dict[var_name] = distribution
        sets_list.append(set_dict)  # Add the generated set to the list
    # Convert the list of dictionaries into a DataFrame
    sets_df = pd.DataFrame(sets_list)
    # Save dataframe as csv file
    monte_carlo_filename = (set_name + '_bioconfig.csv')
    monte_carlo_location = os.path.join(file_dir, monte_carlo_filename)
    sets_df.to_csv(monte_carlo_location, index=False)
    return sets_df


def monte_carlo_fault_events(input_set, event_counts, file_name, output_dir = get_root(), seed=42):
    '''
    Generate fault events based on a Monte Carlo simulation

    Parameters:
    input_set (dict): Nested dictionary with parameters for Monte Carlo simulation of each event
    event_counts (dict): Dictionary with the number of events for each event type

    Returns:
    fault_list (list): List of dictionaries with fault events
    '''
    fault_list = []

    for event_type, params in input_set.items():
        count = event_counts.get(event_type, 1)  # Default to 1 if not specified
        for i in range(count):
            fault_seed = seed + i # use a different, but consistent seed for each distribution sampling
            if event_type == "defect_steambarrier":
                fault = {"event_type": event_type, "t_start": sample_distribution(params["min_t_start"], params["max_t_start"], params["dist_t_start"], seed=fault_seed),
                         "duration": sample_distribution(params["min_duration"], params["max_duration"], params["dist_duration"], seed=fault_seed),
                         "steamflow": sample_distribution(params["min_flow"], params["max_flow"], params["dist_flow"], seed=fault_seed)}
            elif event_type == "steam_in_feed":
                fault = {"event_type": event_type, "t_start": sample_distribution(params["min_t_start"], params["max_t_start"], params["dist_t_start"], seed=fault_seed),
                         "duration": sample_distribution(params["min_duration"], params["max_duration"], params["dist_duration"], seed=fault_seed),
                         "steamflow": sample_distribution(params["min_flow"], params["max_flow"], params["dist_flow"], seed=fault_seed)}
            elif event_type == "airflow_OOC":
                fault = {"event_type": event_type, "offset": sample_distribution(params["min_offset"], params["max_offset"], params["dist_flow"], seed=fault_seed)}
            elif event_type == "OUR_OOC":
                fault = {"event_type": event_type, "offset": sample_distribution(params["min_offset"], params["max_offset"], params["dist_flow"], seed=fault_seed)}
            elif event_type == "blocked_spargers":
                fault = {"event_type": event_type, "t_start": sample_distribution(params["min_t_start"], params["max_t_start"], params["dist_t_start"], seed=fault_seed),
                         "duration": sample_distribution(params["min_duration"], params["max_duration"], params["dist_duration"], seed=fault_seed),
                         "offset": sample_distribution(params["min_offset"], params["max_offset"], params["dist_flow"], seed=fault_seed)}            
                #elif params["constant"] == False:

            fault_list.append(fault)
    file_dir = f"{output_dir}/{file_name}_faultconfig.json"
    with open(file_dir, 'w') as file:
        json.dump(fault_list, file, indent=4)
    return fault_list

def sample_distribution(min, max, dist, seed):
    '''
    Sample value from a variety of distributions.

    Parameters:
    - min: Minimum value of the distribution.
    - max: Maximum value of the distribution.
    - dist: Type of distribution to sample from.
    - seed: Seed for the random number generator.

    Returns:
    - distribution[0]: Sampled value from the distribution.
    '''
    # Initialize numpy random generator
    np.random.seed(seed)
    # Handles case where min and max are swapped (e.g. with negative values)
    if min > max:
        min, max = max, min

    median = (min + max) / 2
    range_val = (max - min) / 2
    if min == max and dist != 'constant':
        raise ValueError("Minimum and maximum values are equal")
    if dist == 'normal':
        distribution = np.random.normal(loc=median, scale=range_val, size=1)
    elif dist == 'uniform':
        distribution = np.random.uniform(low=min, high=max, size=1)
    elif dist == 'exponential':
        rate = 1.0 / median  # Transform median to rate parameter
        distribution = np.random.exponential(scale=1.0/rate, size=1)
    elif dist == 'truncated_normal':
        sd = abs((max - min) / 4)  # Standard deviation assuming 95 % of the data are within the lower and upper bound
        a, b = (min - median) / sd, (max - median) / sd
        distribution = truncnorm(a, b, loc=median, scale=sd).rvs(size=1)
    elif dist == 'binary_truncated_normal':
        pos_neg = np.random.choice([-1, 1]) # decides if the value is positive or negative
        sd = abs((max - min) / 4)
        a, b = (min - median) / sd, (max - median) / sd
        distribution = pos_neg * truncnorm(a, b, loc=median, scale=sd).rvs(size=1)
    elif dist == 'constant':
        distribution = [median]
    else:
        raise ValueError(f"Unknown distribution form '{dist}'")
    return distribution[0]


def generate_mixed_dataset(n_total, init_config, bio_config, fault_counts, fault_config, set_name, dataset_mode = 'All', noise = False, 
                           output_dir = 'Default', progress_bar = False, seed = 42):
    '''
    Performs both Monte Carlo simulations for the biokinetic part of the model and the faults events.
    Makes simulations based on the parameters from the Monte Carlo simulations.

    Parameters:
    - n_total: Number of batches to simulate.
    - init_config: Dictionary with recipe for the batch.
    - bio_config: Dictionary with biokinetic parameters.
    - fault_counts: Dictionary with the number of fault events for each type.
    - fault_config: Dictionary with parameters for the Monte Carlo simulation of fault events.
    - set_name: Name of the set of batches.
    - dataset_mode: Mode of the dataset. Options: 'All', 'Fault_training'. (Default: 'All')
    - output_dir: Directory to save the output CSV files. (Default: 'data/simulation_sets')
    - progress_bar: Boolean to show a progress bar. (Default: False)
    '''

    # Create output directory of necessary
    if output_dir == 'Default':
        set_dir = os.path.join(get_simulation_dir(), set_name)
    else:
        set_dir = os.path.join(output_dir, set_name)
    os.makedirs(set_dir, exist_ok=True)

    # Set directory for summaries of Monte Carlo simulations
    monte_carlo_dir = os.path.join(get_root(), 'data/Monte_Carlo')
    os.makedirs(monte_carlo_dir, exist_ok=True)

    # Define set of online parameters
    online_parameters = ['DO_saturation', 'weight', 'F_set', 'air_L_set', 'air_L', 'kLa', 'OUR']

    # Monte Carlo of biokinetic parameters
    bio_parameter_set = monte_carlo_biokinetic(bio_config, n_total, monte_carlo_dir, set_name, seed=seed)

    # Monte Carlo of fault events
    fault_set = monte_carlo_fault_events(fault_config, fault_counts, set_name, output_dir = monte_carlo_dir, seed=seed)

    fault_iter = iter(fault_set)  # Create an iterator for fault configurations

    # Set noise levels
    airflow_std = 1.42
    weight_std = 0.57

    # Initialize progress bar if requested
    if progress_bar:
        progress = widgets.IntProgress(value=0, min=0, max=len(bio_parameter_set), description='Processing:', bar_style='info', orientation='horizontal')
        display(progress)

    for index, row in bio_parameter_set.iterrows():
        # Create the configuration for this batch
        bio_config = {
            'X0': row['X0'],
            'C': row['C'],
            'Y_SX': row['Y_SX'],
            'Y_SE': row['Y_SE'],
            'Y_SO': row['Y_SO'],
            'Y_SC': row['Y_SC'],
            'Y_XS_true': row['Y_XS_true'],
            'Y_XO_true': row['Y_XO_true'],
            'm_s': row['m_s'],
            'm_o': row['m_o'],
        }

        # Try to get a fault event, if available
        try:
            fault_event = next(fault_iter)
        except StopIteration:
            fault_event = None
        
        # Ensure fault_event is a list and not a single dictionary
        if isinstance(fault_event, dict):
            fault_event = [fault_event]

        # Initialize and run the model
        model = fed_batch_model.A_oryzae(init_config, bio_config, fault_event if fault_event else None)

        if dataset_mode == 'All':
            output = model.control_solve()
        if dataset_mode == 'Fault_training':
            output = model.control_solve(parameters_output=online_parameters, track_state_var=False, target_column=True)
        
        # Add noise for weight and airflow sensor
        if noise == True:
            preprocessing.add_white_noise(output, 'air_L', airflow_std, seed=seed)
            preprocessing.add_white_noise(output, 'weight', weight_std, seed=seed)

        # Save the output to a CSV
        output_file_path = os.path.join(set_dir, f'batch_{index+1}.csv')
        output.to_csv(output_file_path, index=False)

        # Update progress bar if active
        if progress_bar:
            progress.value += 1

    if progress_bar:
        progress.description = 'Done'