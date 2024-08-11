import numpy as np
import pandas as pd
import os
from natsort import natsorted

def load_batchset(data_path):
    '''
    Parameters:
        data_path: Path to folder with batches (saved as individual csv files)

    Returns:
        list of DataFrames: List containing a DataFrame for each batch file
    '''
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"The directory does not exist: {data_path}")
    
    # List all files in the directory
    #files = sorted([os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.csv')])
    files = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.csv')]
    sorted_files = natsorted(files)

    # Load each CSV file into a DataFrame
    batch_set = []
    for file in sorted_files:
        try:
            df = pd.read_csv(file)
            batch_set.append(df)
        except Exception as e:
            raise ValueError(f"Failed to load {file}. Error: {e}")

    return batch_set

class dataloader():
    '''
    Class to load and preprocess batches of data for training
    '''
    def __init__(self, batchset, batch_size=1, seed=42):
        '''
        batchset: list of DataFrames, each DataFrame represents a batch (use load_batchset function to load batches from a directory)
        batch_size: number of batches for each iteration of training
        seed: random seed for reproducibility
        '''
        self.seed = seed
        np.random.seed(self.seed)
        self.batches = batchset
        self.batch_size = batch_size
        self.current_index = 0
        self.total_batches = len(self.batches)
        self.standardized = False
        self.standardize_cols = None

    def shuffle_batches(self):
        '''
        Shuffle the order of batches batches
        '''
        np.random.shuffle(self.batches)

    def combine_batches(self, split_batches):
        '''
        Return a combined dataframe with all batches
        '''
        return pd.concat(split_batches, ignore_index=True)
    
    def airflow_preprocess(self, batch, target_cols):
        '''
        Preprocess airflow data
        Parameters:
        - batch: pd.DataFrame
        - target_cols: list of target columns
        '''
        target_df = batch[target_cols]
        batch.drop(columns=target_cols, inplace=True)
        batch['air_L_diff'] = batch['air_L_set'] - batch['air_L']
        batch['air_past_average'] = batch['air_L_diff'].rolling(window=150, min_periods=1).mean()
        batch.drop(columns=['air_L_set', 'air_L'], inplace=True)
        batch = pd.concat([batch, target_df], axis=1)
        return batch

    def apply_airflow_preprocess(self, target_cols):
        '''
        Use difference between setpoint and actual airflow as well as past average airflow as features
        '''
        for i, batch in enumerate(self.batches):
            self.batches[i] = self.airflow_preprocess(batch, target_cols)

    def combine_fault_events(self, combined_cols, new_col_name):
        '''
        Combine selected fault events into one column
        Parameters:
        - combined_cols (list): list of columns to combine
        - new_col_name (str): name of the new column
        '''
        for i, batch in enumerate(self.batches):
            self.batches[i][new_col_name] = batch[combined_cols].max(axis=1)
            self.batches[i].drop(columns=combined_cols, inplace=True)
            # move 'no_fault' column to the end
            df = self.batches[i].pop('no_fault')
            self.batches[i]['no_fault'] = df

    
    def standardize_data(self, exclude_cols=None):
        '''
        Standardize data by removing the mean and scaling to unit variance

        Parameters:
           - exclude_cols: list of columns to exclude from standardization
        '''
        combined_batches = self.combine_batches(self.batches)

        if exclude_cols is None:
            exclude_cols = []

        self.standardized_cols = combined_batches.columns.difference(exclude_cols)
        if self.standardized is False:
            self.feature_means = combined_batches[self.standardized_cols].mean()
            self.feature_stds = combined_batches[self.standardized_cols].std(ddof=0) 
        combined_batches[self.standardized_cols] = (combined_batches[self.standardized_cols] - self.feature_means) / self.feature_stds
        self.standardized = True
        self.batches = self.split_into_batches(combined_batches, standardized=True)

    def invert_standardization(self, standardized_data):
        '''
        Scale back the standardized data to original scale

        Parameters:
           - data: DataFrame with standardized data

        Returns:
           - list of DataFrames: List of DataFrames with original scale data
        '''
        if not self.standardized:
            raise ValueError("No standardized data available to revert.")
        if isinstance(standardized_data, list):
            original_data = self.combine_batches(standardized_data)
        else:
            original_data = standardized_data.copy()
        original_data[self.standardized_cols] = original_data[self.standardized_cols] * self.feature_stds + self.feature_means
        #self.standardized = False
        if isinstance(standardized_data, list):
            return self.split_into_batches(original_data, standarized=True)
        else:
            return original_data

    def import_standardization(self, dataloader):
        '''
        Import standardization parameters from another dataloader
        '''
        self.standardized = dataloader.standardized
        self.standardized_cols = dataloader.standardized_cols
        self.feature_means = dataloader.feature_means
        self.feature_stds = dataloader.feature_stds

    def split_into_batches(self, combined, standardized=False):
        '''
        Splits the combined DataFrame back into batches.
        Parameters:
            - combined: DataFrame to be split into batches
            - standardized: If True, the time of the DataFrame is standardized and needs to be reverted back to original scale to split into batches
        '''
        # Define tolerance for floating point errors
        tol = 1e-4
        if standardized:
            t = self.invert_standardization(combined)["t"]
        else:
            t = combined["t"]
        # Find the indices after index 0 where 't' resets to 0 indicating a new batch
        reset_points = combined[t.abs() < tol].index
        # Skip the first index as it is the start of the first batch
        reset_points = reset_points[1:]  # Skip the first index
        # Append the end index of the DataFrame to handle the last batch
        reset_points = list(reset_points) + [combined.shape[0]]

        # Split the DataFrame into batches
        batches = []
        start_idx = 0
        for end_idx in reset_points:
            batch = combined.iloc[start_idx:end_idx]
            batches.append(batch)
            start_idx = end_idx

        return batches
    
    def get_data(self, split_batches=True, target_cols=None, separate_target_matrix=False,fuse_target_cols=False):
        '''
        Retrieve data from the dataloader.

        Parameters:
            split_batches: If True, return a list of batches. If False, return a single DataFrame
            target_cols: List of columns to be used as target variables
            separate_target_matrix: If True, return a separate target matrix
            fuse_target_cols: If True, fuse all faults into on target column
        '''
        # if separate_target_matrix:
        #     Y = []
        #     X = []
        #     for batch in self.batches:
        #         Y_batch = batch[target_cols].copy()
        #         if fuse_target_cols:
        #             exclude_col = 'no_fault'
        #             Y_batch['fault'] = Y_batch.drop(columns=exclude_col).max(axis=1)
        #             cols_to_drop = list(set(target_cols) - {exclude_col})
        #             Y_batch = Y_batch.drop(columns=cols_to_drop)
        #         Y.append(Y_batch)
        #         X.append(batch.drop(columns=target_cols))
            
        #     if split_batches:
        #         return X, Y
        #     else:
        #         return self.combine_batches(X), self.combine_batches(Y)
        # else:
        #     if split_batches:
        #         return self.batches
        #     else:
        #         return self.combine_batches(self.batches)
        if separate_target_matrix:
            Y = []
            X = []
            for batch in self.batches:
                Y_batch = batch[target_cols].copy()
                if fuse_target_cols:
                    exclude_col = 'no_fault'
                    Y_batch['fault'] = Y_batch.drop(columns=exclude_col).max(axis=1)
                    cols_to_drop = list(set(target_cols) - {exclude_col})
                    Y_batch = Y_batch.drop(columns=cols_to_drop)
                Y.append(Y_batch)
                X.append(batch.drop(columns=target_cols))
            
            if split_batches:
                return X, Y
            else:
                return self.combine_batches(X), self.combine_batches(Y)
        else:
            XY = []
            for batch in self.batches:
                X_batch = batch.drop(columns=target_cols)
                Y_batch = batch[target_cols].copy()
                if fuse_target_cols:
                    exclude_col = 'no_fault'
                    Y_batch['fault'] = Y_batch.drop(columns=exclude_col).max(axis=1)
                    cols_to_drop = list(set(target_cols) - {exclude_col})
                    Y_batch = Y_batch.drop(columns=cols_to_drop)
                batch
                batch = pd.concat([X_batch, Y_batch], axis=1)
                XY.append(batch)
            if split_batches:
                return XY
            else:
                return self.combine_batches(XY)
        
            

    def train_test_val_split(self, shuffle=True, val_ratio=0.15, test_ratio=0.15):
        '''
        Split the data into training, validation and test sets
        '''
        train_ratio = 1 - val_ratio - test_ratio
        if train_ratio + val_ratio + test_ratio != 1:
            raise ValueError("The sum of train_ratio, val_ratio and test_ratio should be 1")
        if shuffle:
            self.shuffle_batches()
        n_batches = len(self.batches)
        train_batches = int(n_batches * train_ratio)
        val_batches = int(n_batches * val_ratio)

        train_data = self.batches[:train_batches]
        val_data = self.batches[train_batches:train_batches+val_batches]
        test_data = self.batches[train_batches+val_batches:]

        return train_data, val_data, test_data

    def prepare_iterator_data(self, target_cols, one_class=False):
        '''
        Prepare the iterator for training
        '''
        self.one_class_iter = one_class
        self.target_cols_iter = target_cols
        self.data_iterator = self.get_data(split_batches=True, target_cols=target_cols, separate_target_matrix=False, fuse_target_cols=one_class)

    def shuffle_iterator(self):
        '''
        Shuffle the order of batches batches
        '''
        np.random.shuffle(self.data_iterator)
    
    def __iter__(self):
        '''
        Initialize the iterator for training
        '''
        if isinstance(self.data_iterator, type(None)):
            raise ValueError("Data iterator is not initialized. Run prepare_iterator_data method before iterating.")
        self.current_index = 0
        self.shuffle_iterator()  # Shuffle batches at the start of each epoch
        return self

    def __next__(self):
        '''
        Return the next batch of data
        '''
        if self.current_index >= self.total_batches:
            self.shuffle_iterator()
            self.current_index = 0
            raise StopIteration
        end_index = min(self.current_index + self.batch_size, self.total_batches)
        current_batches = self.data_iterator[self.current_index:end_index]
        self.current_index = end_index
        # combine list of dataframe to one dataframe
        current_batches = pd.concat(current_batches, ignore_index=True)
        if self.one_class_iter:
            X = current_batches.drop(columns=["no_fault", "fault"])
            Y = current_batches[["no_fault", "fault"]]
        else:
            X = current_batches.drop(columns=self.target_cols_iter)
            Y = current_batches[self.target_cols_iter]
        return X, Y

def load_batchfolder(set_dir):
    '''
    Loads all batches of a set inside a folder

    Parameters:
    - set_dir: Directory containing CSV files for each batch.
    '''
    # Initialize lists to hold DataFrames and legend strings
    batch_list = []
    legend_list = []

    # List all CSV files in the directory
    for file_name in os.listdir(set_dir):
        if file_name.endswith('.csv'):
            # Construct the full file path
            file_path = os.path.join(set_dir, file_name)
            
            # Read the CSV into a dataframe
            df = pd.read_csv(file_path)
            
            # Append batch to list
            batch_list.append(df)
            
            # Use CSV names as legend
            legend = file_name[:-4]  # Remove the .csv extension
            legend_list.append(legend)

    return batch_list, legend_list

    