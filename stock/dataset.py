import typing
import numpy as np
import pandas as pd
import tensorflow as tf
from etl import engine


#@module: dataset
class StockPrice:

    """
    A class for preparing and processing stock price data for predictive modeling.

    Parameters
    ----------
    symbol : str
        Stock symbol to process.
    n_days_backward : int, optional
        Number of days to look backward for time series (default is 100).
    n_days_forward : int, optional
        Number of days to predict into the future (default is 5).
    batch_size : int, optional
        Size of batches for model input (default is 64).

    Attributes
    ----------
    train_df : pd.DataFrame or None
        DataFrame containing the training dataset.
    validation_df : pd.DataFrame or None
        DataFrame containing the validation dataset.
    test_df : pd.DataFrame or None
        DataFrame containing the testing dataset.
    __full_df : pd.DataFrame or None
        Private attribute for storing the complete dataset.
    """

    def __init__(
        self,
        symbol          : str,
        n_days_backward : int = 100,
        n_days_forward  : int = 5,
        batch_size      : int = 64,
    ):

        self.symbol             = symbol
        self.n_days_backward    = n_days_backward
        self.n_days_forward     = n_days_forward
        self.batch_size         = batch_size

        self.train_df           : typing.Optional[pd.DataFrame] = None
        self.validation_df      : typing.Optional[pd.DataFrame] = None
        self.test_df            : typing.Optional[pd.DataFrame] = None
        self.__full_df          : typing.Optional[pd.DataFrame] = None

    @property
    def full_df(self):

        """
        Retrieves and processes the stock data from a database, and returns it as a DataFrame.

        This property checks if the data is already loaded and cached. If not, it loads data 
        from a database, calculates log returns for the stock price columns, and prepares future 
        close return columns. The data is then cached for future use.

        Returns:
        --------
        pd.DataFrame
            The processed DataFrame containing stock prices, volume, and their log returns.
        """

        # Return cached DataFrame
        if self.__full_df is not None:
            return self.__full_df

        # Load data from the database
        self.__full_df = pd.read_sql(
            f"""
            select 
                "symbol", "date", 
                "open", "high", "low", "close",
                "volume"
            from "daily"
            where "symbol" = '{self.symbol}'
            order by "date" asc
            """,
            con=engine,
            parse_dates={'date': 'YYYY-MM-DD'}
        )
        # Calculate log returns for the stock price and volume columns
        for column in ['open', 'high', 'low', 'close', 'volume']:
            self.__full_df[f'{column}_r'] = np.log(
                self.__full_df[column] / self.__full_df[column].shift(periods=1)
            )
        # Drop any rows with missing values
        self.__full_df = self.__full_df.dropna(axis=0, how='any')
        # Prepare future close return columns
        for s in range(1, self.n_days_forward + 1):
            self.__full_df[f'close_r{s}'] = self.__full_df[f'close_r'].shift(periods=-s)

        return self.__full_df

    # read-only
    @property
    def last_sample(self) -> pd.DataFrame:

        """
        Read-only property to get the last sample of the data.
        
        Returns:
            pd.DataFrame: The last 'n_days_backward' days of data.
        """

        return self.full_df.iloc[
            -self.n_days_backward:, 
            self.full_df.columns.get_indexer(
                ['symbol','date','close','open_r','high_r','low_r','close_r','volume_r']
            )
        ]

    def generate(
        self,
        drop_extremes: bool,
        split_ratios: typing.Tuple[float, float, float],
    ) -> typing.Generator[typing.Optional[tf.data.Dataset], None, None]:

        """
        Generates TensorFlow datasets for training, validation, and testing.

        Parameters
        ----------
        drop_extremes : bool
            Indicates whether to remove outliers from the dataset.
        split_ratios : tuple of float
            Ratios for splitting the dataset into train, validation, and test sets.

        Yields
        ------
        tf.data.Dataset or None
            TensorFlow datasets for each data split.
        """

        # Specify different subsets of columns
        feature_columns = ['open_r','high_r','low_r','close_r','volume_r']
        target_columns = [f'close_r{s}' for s in range(1, self.n_days_forward + 1)]
        selected_columns = ['date'] + feature_columns + target_columns

        # Filter the dataframe to include only the selected columns
        feeding_data = self.full_df[selected_columns]

        # Remove rows with any missing values due to .shift() operation
        feeding_data = feeding_data.dropna(axis=0, how='any')

        # Identify and remove extreme values
        if drop_extremes:
            feeding_data = self.__drop_extremes(
                table=feeding_data, 
                on_columns=feature_columns,
                lower_threshold=0.001,
                upper_threshold=0.999,
            )

        # Split the data into training, validation, and testing sets
        self.train_df, self.validation_df, self.test_df = self.__split(
            table=feeding_data, 
            ratios=split_ratios
        )

        # Define target columns for prediction
        target_columns = [f'close_r{s}' for s in range(1, self.n_days_forward + 1)]

        # Generate TensorFlow datasets for each split
        for df in (self.train_df, self.validation_df, self.test_df):
            if df.empty:
                yield
            else:
                yield tf.keras.utils.timeseries_dataset_from_array(
                    df[['open_r','high_r','low_r','close_r','volume_r']].to_numpy(),
                    targets=df[target_columns].to_numpy(),
                    sequence_length=self.n_days_backward,
                    batch_size=self.batch_size,
                    shuffle=True,
                    seed=42,
                )

    @staticmethod
    def __drop_extremes(
        table: pd.DataFrame,
        on_columns: typing.List[str],
        lower_threshold: float = 0.001, 
        upper_threshold: float = 0.999,
    ) -> pd.DataFrame:

        """
        Removes rows with extreme values based on specified percentiles in a DataFrame.

        Parameters:
        ----------
        table : pd.DataFrame
            The DataFrame from which to remove extreme values.
        on_columns : typing.List[str]
            List of column names to check for extreme values.
        lower_threshold : float, optional
            The lower percentile threshold. Rows with values below this percentile 
            in any of the specified columns will be dropped. Default is 0.001.
        upper_threshold : float, optional
            The upper percentile threshold. Rows with values above this percentile 
            in any of the specified columns will be dropped. Default is 0.999.
        
        Returns:
        ----------
        pd.DataFrame
            A new DataFrame with rows containing extreme values removed.
        """
        # Calculate the upper and lower threshold values for each column
        upper_extremes = table[on_columns].quantile(upper_threshold)
        lower_extremes = table[on_columns].quantile(lower_threshold)
        # Create boolean masks for rows to keep
        is_below_upper = (table[on_columns] < upper_extremes).all(axis=1)
        is_above_lower = (table[on_columns] > lower_extremes).all(axis=1)
        # Filter the table based on these masks
        filtered_table = table[is_below_upper & is_above_lower]

        return filtered_table

    @staticmethod
    def __split(
        table: pd.DataFrame,
        ratios: typing.Tuple[float, float, float],
    ) -> typing.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        """
        Splits a DataFrame into training, validation, and testing sets based on given 
        split ratios. The split is done sequentially, ensuring the temporal order of 
        the data is maintained.

        Parameters:
        ----------
        table : pd.DataFrame
            The DataFrame to be split.
        ratios : typing.Tuple[float, float, float]
            A tuple of three floats representing the proportion of data to be used for 
            training, validation, and testing. The sum of these values should be 1.

        Returns:
        ----------
        typing.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Three DataFrames representing the training, validation, and testing sets.
        """

        # Ensure the split ratios sum up to 1
        ratios = tuple(r / sum(ratios) for r in ratios)
        # Calculate the indices for splitting the dataset
        n_samples = table.shape[0]
        train_end_index = int(n_samples * ratios[0])
        validation_end_index = train_end_index + int(n_samples * ratios[1])
        # Split the data into training, validation, and testing sets
        train_set = table.iloc[:train_end_index]
        validation_set = table.iloc[train_end_index:validation_end_index]
        test_set = table.iloc[validation_end_index:]

        # Return the split datasets
        return train_set, validation_set, test_set



if __name__ == '__main__':
    self = StockPrice('AAPL')
    it = self.generate(drop_extremes=False, split_ratios=(0.8,0.2,0.2))



