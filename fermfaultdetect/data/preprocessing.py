import numpy as np

def add_white_noise(dataframe, column, std, seed=41):
    """
    Creates white noise with the same length as the dataframe and adds it as a new column.
    
    Parameters:
    - dataframe: pandas DataFrame, the data to add the noise to.
    - column: string, the column name to add the noise to.
    - std: float, the standard deviation of the white noise.
    - seed: int, the seed for the random number generator.
    """
    # Set the seed for reproducibility
    np.random.seed(seed)
    # Create the white noise
    noise = np.random.normal(0, std, len(dataframe.index))
    # Add the noise to the dataframe
    dataframe[column] = dataframe[column] + noise
    return dataframe