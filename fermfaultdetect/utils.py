import os

def get_root():
    '''
    Return the root directory of the project.
    '''
    # Get the directory of the current script
    root_dir = os.path.dirname(os.path.abspath(__file__))
    # Move up one directory level
    return os.path.abspath(os.path.join(root_dir, os.pardir))

def get_simulation_dir():
    '''
    Return the directory where simulation data is stored.
    '''
    return os.path.join(get_root(), 'data/simulation_sets')

def get_models_dir():
    '''
    Return the directory where models are stored.
    '''
    return os.path.join(get_root(), 'models')

def get_evaluation_dir():
    '''
    Return the directory where the model evaluations are saved.
    '''
    return os.path.join(get_root(), 'evaluation')

def detection_accuracy(FDR, FAR):
    '''
    Calculate the accuracy of the model based on the False Detection Rate (FDR) and False Alarm Rate (FAR).
    '''
    accuracy = (FDR+(1-FAR))/(FDR+(1-FDR)+FAR+(1-FAR))
    return accuracy