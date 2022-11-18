import os
import pandas as pd
import numpy as np

def initialize_dataset(subject_ids):
    # TODO: Bring over initialization and dataset code
    """
    Initializes dataset from subject_list and files found in data folder.
    
    Parameters
    ----------
    subject_list: list
        List of subject_ids to include in dataset.

    Returns
    -------
    df: pandas.core.frame.DataFrame
        A pandas DataFrame of dataset indexed by subject ids, trial numbers, and labels
    """
    data_list, label_list, subject_list = [], [], []
    filenames = os.listdir(os.getcwd()+"/data/")
    for filename in filenames:
        if not filename.startswith("."):
            subject_id = os.path.basename(filename).strip("sub_.csv")
            if int(subject_id) in subject_ids:
                df = pd.read_csv(os.getcwd()+"/data/"+filename)
                grouped = df.groupby(["chunk","label"])
                grouped_list, labels = [], []
                for tuple, group in grouped:
                    data = group.drop(columns=["chunk","label"]).to_numpy(dtype=np.float32)
                    grouped_list.append(data)
                    labels.append(tuple[1])
                data_list.append(grouped_list)
                label_list.append(labels)
                subject_list.append(np.ones(len(labels))*int(subject_id))
    # construct pandas DataFrame
    data_list, label_list, subject_list = np.array(data_list), np.array(label_list).ravel(), np.array(subject_list).ravel().astype(int)
    newshape = (data_list.shape[0] * data_list.shape[1], data_list.shape[2] * data_list.shape[3])
    data_list = np.reshape(data_list, newshape)
    shifts = np.insert(label_list[1:] == label_list[:-1], 0, True)
    counter = 0
    trial_list = np.zeros(len(shifts)).astype(int)
    for i, shift in enumerate(shifts):
        if not shift: counter = (counter+1)%16
        trial_list[i] = counter+1
    counter = 0
    index_list = np.zeros(len(shifts)).astype(int)
    for i, shift in enumerate(shifts):
        if not shift: counter = 0
        index_list[i] = counter+1
        counter = counter+1
    df = pd.DataFrame(data_list)
    df["index"] = index_list
    df["trial"] = trial_list
    df["subject"] = subject_list
    df["label"] = label_list
    return df