import numpy as np
import pandas as pd
from scipy.stats import chi2
from fermfaultdetect.model_evaluation import detection_accuracy
from fermfaultdetect.data.utils import dataloader
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import wandb
import copy



class pca_fdm:
    def __init__(self, pc, alpha, mw):
        self.pc = pc
        self.alpha = alpha
        self.mw = mw
        self.pca_model = PCA(n_components=self.pc)

    def calibrate(self, X):
        '''
        Calibrates the PCA model using the training data.

        Parameters:
        - X (pd.DataFrame): Training data. Must be one DataFrame with the unfolded batch data.
        '''
        X_pca = self.pca_model.fit_transform(X)

        # Calculate T^2 (scores squared scaled by eigenvalues)
        self.T2 = np.sum((X_pca ** 2) / self.pca_model.explained_variance_, axis=1)

        # Calculate SPE (squared prediction error)
        X_reconstructed = self.pca_model.inverse_transform(X_pca)
        SPE = np.sum((X - X_reconstructed) ** 2, axis=1)

        # Determine the control limits for T^2 and SPE
        self.T2_limit = chi2.ppf(1 - self.alpha, df=self.pca_model.n_components_)
        self.SPE_limit = np.percentile(SPE, 100 * (1 - self.alpha))

    def predict(self, test_X):
        '''
        Predicts fault events using the PCA model.

        Parameters:
        - test_X (pd.DataFrame): Test data. Must be one DataFrame with the unfolded data.
        '''
        # Projecting the test data onto the PCA model
        test_X_pca = self.pca_model.transform(test_X)

        # Calculating SPE and T² for test data
        T2_test = np.sum((test_X_pca ** 2) / self.pca_model.explained_variance_, axis=1)
        X_reconstructed_test = self.pca_model.inverse_transform(test_X_pca)
        SPE_test = np.sum((test_X - X_reconstructed_test) ** 2, axis=1)

        # Using the control limits to detect potential faults
        faults_detected_T2 = T2_test > self.T2_limit
        faults_detected_SPE = SPE_test > self.SPE_limit

        # Initialize pred_test_Y with the length of the faults_detected arrays
        predictions = pd.DataFrame(index=range(len(faults_detected_T2)), columns=["no_fault", "fault"])
        predictions["fault"] = (faults_detected_T2 | faults_detected_SPE).astype(int)
        predictions["no_fault"] = (~predictions["fault"].astype(bool)).astype(int)
        predictions_mw = self.apply_moving_time_window(predictions, self.mw)

        return predictions_mw
    
    def apply_moving_time_window(self, df_in, n):
        """
        Applies a moving time window to the DataFrame to update 'fault' column based on consecutive time points.
        
        Parameters:
        - df_in (pd.DataFrame): Input DataFrame with columns 'no_fault' and 'fault'.
        - n (int): Number of consecutive time points required to trigger a fault event.
        
        Returns:
        - pd.DataFrame: DataFrame with updated 'fault' events based on the moving time window.
        """
        df = df_in.copy()
        # Calculate the rolling sum of 'fault' over the window of size n
        df['temp_fault'] = df['fault'].rolling(window=n).sum().fillna(0).astype(int)

        # Identify the positions where the rolling sum equals n (i.e., all 1s within the window)
        df['temp_trigger'] = df['temp_fault'] == n

        df.drop(columns=["fault"], inplace=True)
        df["fault"] = df["temp_trigger"].astype(int)

        df["no_fault"] = 1 - df["fault"]
        df.drop(columns=["temp_trigger", "temp_fault"], inplace=True)

        return df
    
    def prediction_accuracy(self, test_X, test_Y):
        '''
        Calculate accuracy of the model based on the test data.

        Parameters:
        - test_X (pd.DataFrame): Test features.
        - test_Y (pd.DataFrame): Test labels.

        Returns:
        - float: Detection accuracy of the model.
        '''
        predictions = self.predict(test_X)
        cm = confusion_matrix(1-test_Y["fault"], 1-predictions["fault"], normalize = 'true')
        FDR = cm[0, 0].astype(float)
        FAR = cm[1, 0].astype(float)
        acc = detection_accuracy(FDR, FAR)
        return acc
    
class pls_fdm:
    def __init__(self, n_components, threshold, mw):
        self.pls_model = PLSRegression(n_components=n_components, scale=False, copy=False)
        self.n_components = n_components
        self.threshold = threshold
        self.mw = mw

    def train(self, train_X, train_Y):
        '''
        Trains the PLS model using the training data.
        '''
        self.pls_model.fit(train_X, train_Y)
        self.Y_columns = train_Y.columns

    def predict(self, test_X):
        '''
        Predicts fault events using the PLS model.

        Parameters:
        - test_X (pd.DataFrame): Test data.
        '''
        predictions = self.pls_model.predict(test_X)

        # Format predictions to DataFrame
        predictions = pd.DataFrame(predictions, columns=self.Y_columns)
        # Apply moving window
        predictions_mw = self.apply_moving_time_window(predictions, self.mw, self.threshold)
        return predictions_mw
    
    def apply_moving_time_window(self, predictions, mw, threshold):
        """
        Applies a moving time window to the DataFrame to update 'fault' column based on consecutive time points.
        
        Parameters:
        - df (pd.DataFrame): Input DataFrame with columns 'no_fault' and 'fault'.
        - mw (int): Number of consecutive time points required to trigger a fault event.
        - treshold (float): Threshold for triggering a fault event. Ranges from 0 - 1 (scaled by moving time window)
        
        Returns:
        - pd.DataFrame: DataFrame with updated 'fault' events based on the moving time window.
        """
        df = predictions.copy()
        # Calculate the rolling sum of 'fault' over the window of size n
        df['temp_fault'] = df['fault'].rolling(window=mw).sum().fillna(0).astype(int)

        # Identify the positions where temp_fault if over trigger
        trigger = threshold * mw
        df['temp_trigger'] = df['temp_fault'] >= trigger
        df.drop(columns=["fault"], inplace=True)
        df["fault"] = df["temp_trigger"].astype(int)
        df["no_fault"] = 1 - df["fault"]
        df.drop(columns=["temp_trigger", "temp_fault"], inplace=True)

        return df
    
    def prediction_accuracy(self, test_X, test_Y):
        '''
        Calculate accuracy of the model based on the test data.

        Parameters:
        - test_X (pd.DataFrame): Test features.
        - test_Y (pd.DataFrame): Test labels.

        Returns:
        - float: Detection accuracy of the model.
        '''
        predictions = self.predict(test_X)
        cm = confusion_matrix(1-test_Y["fault"], 1-predictions["fault"], normalize = 'true')
        FDR = cm[0, 0].astype(float)
        FAR = cm[1, 0].astype(float)
        acc = detection_accuracy(FDR, FAR)
        return acc

    def clear_large_attributes(self):
        '''
        Clear large internal attributes that are not needed for prediction. Avoids large pickle files.
        '''
        if hasattr(self.pls_model, '_x_scores'):
            self.pls_model._x_scores = None
        if hasattr(self.pls_model, '_y_scores'):
            self.pls_model._y_scores = None
        if hasattr(self.pls_model, 'x_scores_'):
            self.pls_model.x_scores_ = None
        if hasattr(self.pls_model, 'y_scores_'):
            self.pls_model.y_scores_ = None

    
class svm_fdm:
    def __init__(self, kernel, gamma, C, mw, seed=42):
        '''
        Initializes the SVM fault detection model model with the specified parameters.

        Parameters:
        - kernel (str): Type of kernel to be used ('rbf', 'poly', and 'sigmoid').
        - gamma (float): Kernel coefficient.
        - C (float): Penalty parameter C of the error term.
        - mw (int): Number of consecutive time points required to trigger a fault event.
        '''
        self.mw = mw
        self.svm_model = SVC(kernel=kernel, gamma=gamma, C=C, random_state=seed)

    def train(self, train_X, train_Y):
        '''
        Trains the SVM model using the training data.
        '''
        self.svm_model.fit(train_X, train_Y)

    def predict(self, test_X):
        '''
        Predicts fault events using the SVM model.

        Parameters:
        - test_X (pd.DataFrame): Test data.

        Returns:
        - pd.DataFrame: Updated predictions based on the moving time window.
        '''
        predictions = self.svm_model.predict(test_X)
        predictions = pd.DataFrame(predictions, columns=["fault"])
        predictions_mw = self.apply_moving_time_window(predictions, self.mw)
        return predictions_mw
    
    def change_mw(self, mw):
        '''
        Change the moving time window of the model
        Parameters:
        - mw (int): New moving time window
        '''
        self.mw = mw

    def apply_moving_time_window(self, predictions, mw):
        """
        Applies a moving time window to the DataFrame to update 'fault' column based on consecutive time points.
        
        Parameters:
        - df (pd.DataFrame): Input DataFrame with columns 'no_fault' and 'fault'.
        - mw (int): Number of consecutive time points required to trigger a fault event.
        
        Returns:
        - pd.DataFrame: DataFrame with updated 'fault' events based on the moving time window.
        """
        df = predictions.copy()
        # Calculate the rolling sum of 'fault' over the window of size n
        df['temp_fault'] = df['fault'].rolling(window=mw).sum().fillna(0).astype(int)

        # Identify the positions where temp_fault if over trigger
        df['temp_trigger'] = df['temp_fault'] == mw
        df.drop(columns=["fault"], inplace=True)
        df["fault"] = df["temp_trigger"].astype(int)
        df["no_fault"] = 1 - df["fault"]
        df.drop(columns=["temp_trigger", "temp_fault"], inplace=True)

        return df
    
    def prediction_accuracy(self, test_X, test_Y):
        '''
        Calculate accuracy of the model based on the test data.

        Parameters:
        - test_X (pd.DataFrame): Test features.
        - test_Y (pd.DataFrame): Test labels.

        Returns:
        - float: Detection accuracy of the model.
        '''
        predictions = self.predict(test_X)
        cm = confusion_matrix((1-test_Y), (1-predictions["fault"]), normalize = 'true')
        FDR = cm[0, 0]
        FAR = cm[1, 0]
        acc = detection_accuracy(FDR, FAR)
        return acc
    

class Net(nn.Module):
    '''
    Basic class for creating an FFNN model.
    '''
    def __init__(self, input_dim, output_dim, hidden_layers):
        super(Net, self).__init__()
        layers = []

        # Input layer
        current_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.linear_relu_stack = nn.Sequential(*layers)
        
        # Apply initialization to the layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.linear_relu_stack(x)
    
    
class CrossEntropyLossPenalty(torch.nn.Module):
    def __init__(self, penalty_class=None, penalty_factor=1.0):
        """
        Initialize the custom loss function.

        Args:
            penalty_class (int, optional): The class index to penalize more if predicted incorrectly. Defaults to None.
            penalty_factor (float, optional): The factor by which to increase the loss for the penalty_class. Defaults to 1.0.
        """
        super(CrossEntropyLossPenalty, self).__init__()
        self.penalty_class = penalty_class
        self.penalty_factor = penalty_factor

    def forward(self, input, target):
        """
        Compute the custom cross entropy loss.

        Args:
            input (torch.Tensor): Predictions from the model (logits, not probabilities).
            target (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Computed custom cross entropy loss.
        """
        # Standard cross entropy loss
        loss = F.cross_entropy(input, target, reduction='none')
        
        if self.penalty_class is not None and self.penalty_factor > 1.0:
            # Identify the predictions that are penalized
            incorrect_predictions = (torch.argmax(input, dim=1) != target) & (target == self.penalty_class)
            
            # Apply the penalty factor to the incorrect predictions
            loss[incorrect_predictions] *= self.penalty_factor

        # Return the mean loss, since the loss is normally averaged by default
        return loss.mean()
    

class nn_fdm:
    def __init__(self, input_dim, output_dim, hidden_layers, mode = "one_class", mw=1, seed=42, use_wandb=False):
        '''
        Initializes the neural network fault detection model with the specified parameters.

        Parameters:
        - input_dim (int): Number of input features.
        - output_dim (int): Number of output features.
        - hidden_layers (list): List specifying number of nodes for each layer.
        - mode (str): Choose if individual faults should be identified ("one_class" or "mult_class").
        - mw (int): Number of consecutive time points required to trigger a fault event.
        - seed (int): Random seed for reproducibility.
        - use_wandb (bool): If True, logs the model to Weights & Biases.
        '''
        torch.manual_seed(seed)
        self.mode = mode
        self.use_wandb = use_wandb
        self.mw = mw
        self.net = Net(input_dim, output_dim, hidden_layers)
        if self.use_wandb:
            wandb.watch(self.net, log="all")

    def train(self, train_dl, test_X, test_Y, epochs, batch_size, learning_rate, weight_decay, penalty_class, false_alarm_penalty=1.0, early_stopping=None, print_progress=True):
        '''
        Train the neural network model.

        Parameters:
        - train_dl (DataLoader): Training data in dataloader prepared as an iterator.
        - test_X (pd.DataFrame): Test data.
        - test_Y (pd.DataFrame): Test labels.
        - n_batches_test (int): Number of batches for the test data.
        - epochs (int): Number of epochs to train the model.
        - batch_size (int): Number of batches per training cycle.
        - learning_rate (float): Learning rate for the optimizer.
        - weight_decay (float): Weight decay for the optimizer.
        - penalty_class (int): Class index to penalize more if predicted incorrectly.
        - false_alarm_penalty (float): Factor by which to increase the loss for the penalty_class.
        - early_stopping (int): Number of epochs to wait before stopping training if no improvement is seen.
        - print_progress (bool): If True, prints the progress of the training.
        '''
        # Checking input data types
        if not isinstance(test_X, pd.DataFrame):
            print("test_X should be a torch.Tensor")
            return
        if not isinstance(test_Y, pd.DataFrame):
            print("test_Y should be a torch.Tensor")
            return
        if not isinstance(train_dl, dataloader):
            print("train_dl should be a DataLoader object")
            return
        
        self.penalty_class = penalty_class
        self.target_cols = test_Y.columns
        # transform test data to tensors
        test_X_tensor = torch.tensor(test_X.values, dtype=torch.float32)
        test_Y_tensor = torch.tensor(test_Y.values, dtype=torch.float32).argmax(dim=1)

        # Initialize the optimizer
        optimizer = optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Initialize the loss function
        loss_function = CrossEntropyLossPenalty(penalty_class=self.penalty_class, penalty_factor=false_alarm_penalty)
        #loss_function = nn.CrossEntropyLoss()

        # Setup for early stopping
        best_loss = 0
        best_model_state = copy.deepcopy(self.net.state_dict())
        best_epoch = 0
        epochs_no_improve = 0
        best_metrics = {}

        # setting up lists for handling loss/accuracy
        cur_loss = 0
        train_losses = []
        test_losses = []

        # Number of train batches
        n_train_batches = len(train_dl.data_iterator)

        for epoch in range(epochs):
            # Forward -> Backprob -> Update params
            ## Train
            cur_loss = 0
            self.net.train()
            for X, Y in train_dl:
                # Zero the gradients for all optimizers
                optimizer.zero_grad()
                X = torch.tensor(X.values, dtype=torch.float32)
                Y = torch.tensor(Y.values, dtype=torch.float32).argmax(dim=1)                

                output = self.net(X)  # Remove the batch dimension for FFNN
                batch_loss = loss_function(output, Y)

                batch_loss.backward()

                #update net
                optimizer.step()

                cur_loss += batch_loss.detach().numpy()
            
            train_losses.append(cur_loss / n_train_batches * batch_size)

            # self.net.eval()
            # acc, FDR, FAR  = self.prediction_accuracy(self.net(test_X_tensor), test_Y)
            # # Test loss
            # test_loss = loss_function(self.net(test_X_tensor), test_Y_tensor.argmax(dim=1)).detach().numpy()
            # test_losses.append(test_loss)

            # Evaluation phase
            self.net.eval()

            test_output = self.net(test_X_tensor)
            test_loss = loss_function(test_output, test_Y_tensor).detach().numpy()

            acc, FDR, FAR = self.prediction_accuracy(test_output, test_Y)
            test_losses.append(test_loss)

            if self.use_wandb:
                wandb.log({"loss": train_losses[-1], "accuracy": acc, "FDR": FDR, "FAR": FAR})

            if epoch == 1:
                best_loss = test_loss

            if test_loss <= best_loss:
                best_loss = test_loss
                best_model_state = copy.deepcopy(self.net.state_dict())
                best_epoch = epoch
                best_metrics = {'accuracy': acc, 'FDR': FDR, 'FAR': FAR}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if early_stopping is not None:
                if epochs_no_improve >= early_stopping:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

            if (epoch % 1 == 0) & print_progress==True:
                print(
                    f"Epoch {epoch+1} : Train loss {np.round(train_losses[-1].item(), decimals=4)} , "
                    f"Test loss {np.round(test_losses[-1].item(), decimals=4)} ,"
                    f"Accuracy: {acc:.3f} , "
                    f"FDR: {FDR:.3f} , "
                    f"FAR: {FAR:.3f}"
            )

        self.net.load_state_dict(best_model_state)
        print(f"Best Epoch: {best_epoch + 1} with Accuracy: {best_metrics['accuracy']:.3f}, FDR: {best_metrics['FDR']:.3f}, FAR: {best_metrics['FAR']:.3f}")
                
        epoch = np.arange(len(train_losses))
        if print_progress==True:
            plt.figure()
            plt.plot(epoch, train_losses, 'r', label='Train Loss')
            plt.plot(epoch, test_losses, 'b', label='Test Loss')
            plt.legend()
            plt.xlabel('Updates'), plt.ylabel('Loss')

    def predict(self, test_X, output_columns=None):
        '''
        Return predictions from the neural network model with moving time window applied (if specified in instructor).
        Parameters:
        - test_X (pd.DataFrame or torch.Tensor): Test data.
        '''
        self.net.eval()
        if isinstance(test_X, pd.DataFrame):
            test_X = torch.tensor(test_X.values, dtype=torch.float32)
        pred_tensor = self.net(test_X)
        predictions = self.output_to_prediction_df(pred_tensor, output_columns)
        predictions_mw = self.apply_moving_time_window(predictions, self.mw)
        return predictions_mw

    def output_to_prediction_df(self, pred_tensor, output_columns=None):
        '''
        Convert the output tensor from the model to a DataFrame with the correct labels.

        Parameters:
        - pred_tensor (torch.Tensor): Predictions from the model.
        - output_columns (list): List of names for labels for the predictions dataframe.

        Returns:
        - pd.DataFrame: Predictions.
        '''
        if output_columns is not None:
            self.target_cols = output_columns
        pred_tensor = F.softmax(pred_tensor, dim=1)
        # Transform softmax into one hot encoded predictions
        _, pred_classes = torch.max(pred_tensor, dim=1)
        predictions = torch.zeros_like(pred_tensor)
        predictions.scatter_(1, pred_classes.unsqueeze(1), 1)
        predictions = predictions.detach().numpy()
        if self.mode == "one_class":
            predictions_df = pd.DataFrame(predictions, columns=["no_fault", "fault"]).astype(int)
        elif self.mode == "mult_class":
            predictions_df = pd.DataFrame(predictions, columns=self.target_cols).astype(int)
        return predictions_df

    def prediction_accuracy(self, pred_tensor, test_Y):
        '''
        Calculate accuracy, false detection and false alarm rate of the model based on the test data.
        Parameters:
        - pred_tensor (torch.Tensor): Predictions from the model.
        - test_Y (pd.DataFrame): Test labels.
        Returns:
        - float: Detection accuracy of the model.
        - float: False detection rate of the model.
        - float: False alarm rate of the model.
        '''
        pred_tensor = F.softmax(pred_tensor, dim=1)
        # transform softmax into one hot encoded predictions
        _, pred_classes = torch.max(pred_tensor, dim=1)
        predictions = torch.zeros_like(pred_tensor)
        predictions.scatter_(1, pred_classes.unsqueeze(1), 1)
        predictions = predictions.detach().numpy()
        predictions_df = pd.DataFrame(predictions, columns=test_Y.columns).astype(int)
        pred_Y = predictions_df.drop(columns="no_fault").max(axis=1).astype(int)
        if self.mode == "mult_class":
            test = test_Y.drop(columns="no_fault").max(axis=1).astype(int)
            cm = confusion_matrix(1-test, 1-pred_Y, normalize = 'true')
        elif self.mode == "one_class":
            cm = confusion_matrix(1-test_Y["fault"], 1-pred_Y, normalize = 'true')
        FDR = cm[0, 0].astype(float)
        FAR = cm[1, 0].astype(float)
        acc = detection_accuracy(FDR, FAR)
        return acc, FDR, FAR
    

    def apply_moving_time_window(self, predictions, mw):
        """
        Applies a moving time window to the DataFrame to update columns based on consecutive time points.
        
        Parameters:
        - predictions (pd.DataFrame): Input DataFrame with various columns.
        - mw (int): Number of consecutive time points required to trigger an event.
        - excluded_columns (list): List of columns to exclude from the moving time window application.
        
        Returns:
        - pd.DataFrame: DataFrame with updated events based on the moving time window.
        """
        df = predictions.copy()
        fault_columns = []

        # Drop the fault columns to avoid applying the moving time window to them
        df.drop(columns=["no_fault"], inplace=True)

        # Iterate over each column target column
        for column in df.columns:
            # Calculate the rolling sum over the window of size mw
            df[f'temp_{column}'] = df[column].rolling(window=mw).sum().fillna(0).astype(int)
            # Identify the positions where temp_column is over the trigger
            df[f'temp_trigger_{column}'] = df[f'temp_{column}'] == mw
            df.drop(columns=[column], inplace=True)
            df[column] = df[f'temp_trigger_{column}'].astype(int)
            df.drop(columns=[f'temp_trigger_{column}', f'temp_{column}'], inplace=True)
            fault_columns.append(column)
        
        # Set no_fault column based on the fault columns
        df["no_fault"] = 1 - df[fault_columns].max(axis=1)

        return df

    def change_mw(self, mw):
        '''
        Change the moving time window of the model
        Parameters:
        - mw (int): New moving time window
        '''
        self.mw = mw

    def save_model(self, path):
        '''
        Save the model to a specified path.
        Parameters:
        - path (str): Path to save the model.
        '''
        torch.save(self.net.state_dict(), path)

    def load_model(self, path):
        '''
        Load the model from a specified path.
        Parameters:
        - path (str): Path to load the model.
        '''
        self.net.load_state_dict(torch.load(path))
        self.net.eval()