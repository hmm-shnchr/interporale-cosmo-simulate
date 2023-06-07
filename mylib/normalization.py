import numpy as np
import copy as cp


class Normalization:
    """
    Normalize the dataset.
    The format to be normalized is defined by norm_format.
    Run run() to normalize the dataset.
    Run inv_run() to restore the normalized dataset.
    """

    def __init__(self, norm_format):
        self.norm_format                = norm_format
        self.eps                        = 1e-7
        self.data_min, self.data_max    = None, None
        self.mean, self.std             = None, None
        
    def run(self, dataset):
        dataset_normed                                              = None
        if self.norm_format == "None":              dataset_normed  = self.__none(cp.deepcopy(dataset))
        if self.norm_format == "Normalization":     dataset_normed  = self.__normalize(cp.deepcopy(dataset))
        if self.norm_format == "Standardization":   dataset_normed  = self.__standardize(cp.deepcopy(dataset))
        return dataset_normed
    
    def inv_run(self, dataset_normed):
        dataset_inv                                             = None
        if self.norm_format == "None":              dataset_inv = self.__none(cp.deepcopy(dataset_normed))
        if self.norm_format == "Normalization":     dataset_inv = self.__inv_normalize(cp.deepcopy(dataset_normed))
        if self.norm_format == "Standardization":   dataset_inv = self.__standardize(cp.deepcopy(dataset_normed))
        return dataset_inv

    def run_predict(self, dataset):
        dataset_normed                                             = None
        if self.norm_format == "None":              dataset_normed = self.__none(cp.deepcopy(dataset))
        if self.norm_format == "Normalization":     dataset_normed = self.__normalize(cp.deepcopy(dataset))
        if self.norm_format == "Standardization":   dataset_normed = self.__standardize_predict(cp.deepcopy(dataset))
        return dataset_normed

    def inv_run_predict(self, dataset_normed):
        dataset_inv                                             = None
        if self.norm_format == "None":              dataset_inv = self.__inv_none(cp.deepcopy(dataset_normed))
        if self.norm_format == "Normalization":     dataset_inv = self.__inv_normalize(cp.deepcopy(dataset_normed))
        if self.norm_format == "Standardization":   dataset_inv = self.__inv_standardize_predict(cp.deepcopy(dataset_normed))
        return dataset_inv
        
    def __normalize(self, dataset):
        data_min        = dataset.min()
        data_max        = dataset.max()
        dataset         = (dataset - data_min) / (data_max - data_min) + self.eps
        self.data_min   = data_min
        self.data_max   = data_max
        print("min : {:.3e}, max : {:.3e}".format(data_min, data_max))
        return dataset
    
    def __inv_normalize(self, dataset):
        dataset *= (self.data_max - self.data_min)
        dataset += self.data_min
        dataset -= self.eps
        return dataset
    
    def __standardize(self, dataset):
        mean        = dataset.mean()
        std         = dataset.std()
        dataset     = (dataset - mean) / std
        self.mean   = mean
        self.std    = std
        print("mean : {:.3e}, std : {:.3e}".format(mean, std))
        return dataset
    
    def __inv_standardize(self, dataset):
        dataset *= self.std
        dataset += self.mean
        return dataset

    def __standardize_predict(self, dataset):
        dataset = (dataset - self.mean) / self.std
        return dataset

    def __inv_standardize_predict(self, dataset):
        return self.__inv_standardize(dataset)
    
    def __none(self, dataset):
        return dataset
    
    def __inv_none(self, dataset):
        return dataset