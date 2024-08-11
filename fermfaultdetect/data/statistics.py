import pandas as pd


def aggregate_stats(batch_list):
    """
    Computes the mean and standard deviation for each parameter across a list of DataFrames.
    
    Args:
    dataframes (list of pd.DataFrame): List of DataFrames where each DataFrame has the same parameters as columns
                                      and the index represents time points.
    
    Returns:
    pd.DataFrame: DataFrame containing the mean and standard deviation for each parameter across the given DataFrames.
    """
    # Concatenate all dataframes along columns, forming a multi-level column DataFrame
    combined_df = pd.concat(batch_list, axis=1, keys=range(len(batch_list)))
    
    # Compute mean and standard deviation across the DataFrames for each parameter
    mean_df = combined_df.groupby(level=1, axis=1).mean()
    std_df = combined_df.groupby(level=1, axis=1).std()
    #quantil_df = combined_df.groupby(level=1, axis=1).quantile([0, 1])
    min_df = combined_df.groupby(level=1, axis=1).min()
    max_df = combined_df.groupby(level=1, axis=1).max()
    
    # Prepare the output DataFrame
    output_columns = pd.MultiIndex.from_product([mean_df.columns, ['mean', 'std', 'min', 'max']])
    output_df = pd.DataFrame(index=mean_df.index, columns=output_columns)
    
    for param in mean_df.columns:
        output_df[(param, 'mean')] = mean_df[param]
        output_df[(param, 'std')] = std_df[param]
        output_df[(param, 'min')] = min_df[param]
        output_df[(param, 'max')] = max_df[param]
    
    return output_df