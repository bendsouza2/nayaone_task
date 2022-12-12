import pandas as pd
import numpy as np
from scipy import stats
import os


class Dataset:
    """Class to load and analyse data

    Attributes:
        file_path(str): The file path of the dataset
        data(dataframe): The dataset loaded into a pandas dataframe"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)

    def std_dev(self, column):
        """Returns the standard deviation for a particular column in the dataset

        Args:
            column(str): The name of the column on which to perform the calculation
        """
        deviation = self.data[column].std()
        return deviation

    def mean(self, column):
        """Returns the mean for a particular column in the dataset

                Args:
                    column(str): The name of the column on which to perform the calculation
                """
        avg = self.data[column].mean()
        return avg

    def all_outliers(self):
        """Returns a list of rows that are outliers in the dataset.
        An outlier is defined here as any value for which the z-score of the value against the column is greater than 3.
        If any of the numerical values in the row are outliers, the index for that row is returned.
                """
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numerics_only = self.data.select_dtypes(include=numerics)
        no_outliers = numerics_only[(np.abs(stats.zscore(numerics_only, nan_policy='omit')).fillna(0) < 3).all(
            axis=1)].index
        outliers = list(numerics_only.drop(no_outliers).index)
        return outliers

    def col_outliers(self, col_name):
        """Returns a list of rows that are outliers for that particular column.
                An outlier is defined here as any value for which the z-score of the value against is
                greater than 3.

        Args:
                col_name(str): The name of the column on which to perform the calculation
                        """
        no_outliers = self.data[(np.abs(stats.zscore(self.data[col_name], nan_policy='omit')).fillna(0) < 3)].index
        outliers = list(self.data.drop(no_outliers).index)
        return outliers


