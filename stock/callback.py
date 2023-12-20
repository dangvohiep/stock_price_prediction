import json
import datetime as dt
import numpy as np
import sqlalchemy
import tensorflow as tf

class CustomLog(tf.keras.callbacks.Callback):
    """
    A custom callback class for TensorFlow's Keras API to log training metrics.

    This class is designed to print training metrics in a less verbose manner than the default
    TensorFlow logger, providing clear, periodic updates on the model's performance during training.
    """

    def on_epoch_end(
        self, 
        epoch: int, 
        logs: dict = None
    ):
        """
        Called at the end of an epoch during model training.

        Parameters:
        ----------
        epoch : int
            The index of the epoch.
        logs : dict
            A dictionary containing training metrics.
        """
        # Check if it's time to log (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}:")
            # Iterate through and print each metric
            for key, value in logs.items():
                # Format and align the metric key and value
                print(f"{key.ljust(25)}: {value:<.9f}")
            # Print a separator for readability
            print('--------------------')


class SavePrediction(tf.keras.callbacks.Callback):
    def __init__(
        self, 
        symbol: str,
        model: tf.keras.Model,
        n_days_backward: int,
        n_days_forward: int,
        db_engine: sqlalchemy.engine.base.Engine,
    ):
        super().__init__()
        self.symbol = symbol
        self.model = model
        self.n_days_backward = n_days_backward
        self.n_days_forward = n_days_forward
        self.db_engine = db_engine

    def on_predict_end(self, epoch, logs=None):
        
        with self.db_engine.begin() as db_connection:
            prediction_table = sqlalchemy.Table(
                'prediction',
                sqlalchemy.MetaData(), 
                autoload_with=self.db_engine
            )
            try:    # .predict() called for the first time
                db_connection.execute(
                    statement=sqlalchemy.insert(table=prediction_table), 
                    parameters=[{
                        'date'      : dt.date.today(),
                        'config'    : self.model.to_json(),
                        'symbol'    : self.symbol,
                        'n_days_backward': self.n_days_backward,
                        'n_days_forward' : self.n_days_forward,
                        'value': json.dumps(self.prediction.tolist()),
                    }],
                )
            except sqlalchemy.exc.IntegrityError: # .predict() called from the second time
                # do nothing
                return


    def on_predict_batch_end(self, batch, logs=None):
        # Store the predictions of the last batch
        self.prediction: np.array = logs['outputs'].reshape(-1)

