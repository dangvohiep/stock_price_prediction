import abc
import typing
import numpy as np
import pandas as pd
import tensorflow as tf
import dataset
import callback
import etl


class BaseModel(abc.ABC):

    """
    Base class for different deep learning models for stock price prediction.

    This class provides common functionalities / interfaces for various models to handle
    configuring, training, evaluation, and prediction for time series data.
    """

    def __init__(self) -> None:
        # Placeholder for the model instance
        self.model: typing.Optional[tf.keras.Model] = None

    def evaluate(
        self, 
        test_set: tf.data.Dataset
    ) -> list:
        """
        Evaluate the model on an unseen dataset.

        Parameters:
        ----------
        test_set : tf.data.Dataset
            The dataset on which to evaluate the model.

        Returns:
        -------
        list
            The evaluation result of the model.
        """
        return self.model.evaluate(x=test_set)

    def predict(
        self, 
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Use the model to predict on new data.

        Parameters:
        ----------
        data : pd.DataFrame
            Input data for making predictions.

        Returns:
        -------
        pd.DataFrame
            The DataFrame containing predicted values.
        """
        # Predict returns using the model
        predicted_returns = self.model.predict(
            x=data[['open_r','high_r','low_r','close_r','volume_r']].to_numpy()[np.newaxis,...],
            batch_size=1,
            callbacks=[callback.SavePrediction(
                symbol=set(data['symbol']).pop(),
                model=self.model,
                n_days_backward=data.shape[0],
                n_days_forward=self.model.output_shape[-1],
                db_engine=etl.engine,
            )]
        ).reshape(-1,)
        # Calculate predicted prices from returns
        last_price = data.iloc[-1, data.columns.get_loc('close')]
        last_date = data.iloc[-1, data.columns.get_loc('date')]
        predicted_prices = [last_price]
        for r in predicted_returns:
            predicted_prices.append(np.exp(r) * predicted_prices[-1])
        predicted_prices.pop(0)
        # Prepare DataFrame for predicted data
        predicted_dates = pd.date_range(
            start=last_date, 
            periods=len(predicted_prices) + 1, 
            freq=pd.tseries.offsets.BDay()
        ).date.tolist()[1:]
        predicted_data = pd.DataFrame(
            data={
                'date': predicted_dates, 
                'type': 'predicted', 
                'price': predicted_prices
            }
        ).astype({'date': 'datetime64[ns]'})
        # Combine actual and predicted data
        actual_data = data[['date','close']].rename({'close': 'price'}, axis=1)
        actual_data.insert(loc=1, column='type', value='actual')
        
        return pd.concat(
            [actual_data, predicted_data], axis=0
        ).reset_index(drop=True)

    def fit(
        self, 
        train_set: tf.data.Dataset, 
        validation_set: tf.data.Dataset, 
        max_epochs: int, 
        learning_rate: float
    ) -> tf.keras.callbacks.History:

        """
        Fits the model to the training data.

        Parameters:
        ----------
        train_set : tf.data.Dataset
            The training dataset.
        validation_set : tf.data.Dataset
            The validation dataset.
        max_epochs : int
            The maximum number of epochs for training.
        learning_rate : float
            The learning rate for the optimizer.

        Returns:
        -------
        tf.keras.callbacks.History
            The History object containing training and validation loss and metrics values.
        """
        # Prepare for model building
        first_input_tensor, first_output_tensor = next(iter(train_set))
        feature_dataset = train_set.map(lambda features, labels: features)
        # Build and compile the model if not already done
        normalizer_name = 'normalizer'
        if self.model is None:
            normalizer = tf.keras.layers.Normalization(axis=-1, name=normalizer_name)
            normalizer.adapt(feature_dataset)
            self._build_model(
                input_shape=first_input_tensor.shape[-2:], 
                output_units=first_output_tensor.shape[-1], 
                normalizer=normalizer
            )
            self.model.compile(
                optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate),
                loss=tf.keras.losses.MeanSquaredError(), 
                metrics=[tf.keras.metrics.MeanAbsoluteError()]
            )
        elif normalizer_name in [layer.name for layer in self.model.layers]:
            # Adapt the normalization layer to the training data
            normalization_layer = self.model.get_layer(name=normalizer_name)
            normalization_layer.reset_state()
            normalization_layer.adapt(feature_dataset)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            start_from_epoch=32,
            restore_best_weights=True,
        )
        # Fit the model to the training data
        return self.model.fit(
            train_set,
            epochs=max_epochs,
            validation_data=validation_set,
            callbacks=
                [early_stopping, callback.CustomLog()] 
                if validation_set is not None 
                else [callback.CustomLog()],
            verbose=0,
        )

    @abc.abstractmethod
    def _build_model(
        self, 
        input_shape: typing.Tuple[int, int], 
        output_units: int,
        normalizer: tf.keras.layers.Normalization,
    ) -> None:
    
        """
        Abstract method to build the model. This method must be implemented by subclasses.
        It defines the architecture of the model, including layers and their configurations.

        Parameters:
        ----------
        input_shape : Tuple[int, int]
            The shape of the input data, excluding the batch size. This helps in defining 
            the input layer of the model.
        output_units : int
            The number of units in the output layer, determining the shape of the output 
            produced by the model.
        normalizer : tf.keras.layers.Layer
            A normalization layer that can be applied to the input data. This layer is 
            responsible for standardizing or normalizing the input features.

        Returns:
        -------
        None
            The method does not return anything but must set the `self.model` attribute 
            with the constructed model.
        """
        pass


class NaiveModel(BaseModel):

    # implement
    def _build_model(
        self, 
        input_shape: typing.Tuple[int, int], 
        output_units: int,
        normalizer: tf.keras.layers.Normalization,
    ) -> None:
        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        # compute the average of all features over all time points
        x = tf.keras.layers.GlobalAveragePooling1D()(inputs)
        # get the average of the 3rd feature (close_r)
        x = tf.keras.layers.Lambda(lambda x: x[:, 3:4])(x)
        # propagate the same prediction over all output units
        outputs = tf.keras.layers.Lambda(
            lambda x: tf.repeat(x, repeats=output_units, axis=1)
        )(x)
        self.model = tf.keras.Model(inputs, outputs)


class MLP(BaseModel):

    # implement
    def _build_model(
        self, 
        input_shape: typing.Tuple[int, int], 
        output_units: int,
        normalizer: tf.keras.layers.Normalization,
    ) -> None:
        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        x = normalizer(inputs)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=64, activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.Dense(units=32, activation=tf.keras.activations.relu)(x)
        # no ReLu activation because return can below 0
        outputs = tf.keras.layers.Dense(units=output_units)(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)


class Conv1D(BaseModel):

    # implement
    def _build_model(
        self, 
        input_shape: typing.Tuple[int, int], 
        output_units: int,
        normalizer: tf.keras.layers.Normalization,
    ) -> None:
        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        x = normalizer(inputs)
        x = tf.keras.layers.Conv1D(filters=16, kernel_size=10, activation="relu")(x)
        x = tf.keras.layers.Conv1D(filters=32, kernel_size=10, activation="relu")(x)
        x = tf.keras.layers.Conv1D(filters=64, kernel_size=10, activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        # no ReLu activation because return can below 0
        outputs = tf.keras.layers.Dense(units=output_units)(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)


class LSTM(BaseModel):

    # implement
    def _build_model(
        self, 
        input_shape: typing.Tuple[int, int], 
        output_units: int,
        normalizer: tf.keras.layers.Normalization,
    ) -> None:
        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        x = normalizer(inputs)
        x = tf.keras.layers.LSTM(units=16, activation=tf.keras.activations.tanh, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(units=16, activation=tf.keras.activations.tanh, return_sequences=False)(x)
        # no ReLu activation because return can below 0
        outputs = tf.keras.layers.Dense(units=output_units)(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        

class GRU(BaseModel):

    # implement
    def _build_model(
        self, 
        input_shape: typing.Tuple[int, int], 
        output_units: int,
        normalizer: tf.keras.layers.Normalization,
    ) -> None:
        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        x = normalizer(inputs)
        x = tf.keras.layers.GRU(units=16, activation=tf.keras.activations.tanh, return_sequences=True)(x)
        x = tf.keras.layers.GRU(units=16, activation=tf.keras.activations.tanh, return_sequences=False)(x)
        # no ReLu activation because return can below 0
        outputs = tf.keras.layers.Dense(units=output_units)(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)


class Bidirectional(BaseModel):

    # implement
    def _build_model(
        self, 
        input_shape: typing.Tuple[int, int], 
        output_units: int,
        normalizer: tf.keras.layers.Normalization,
    ) -> None:
        # Define model
        inputs = tf.keras.Input(shape=input_shape)
        x = normalizer(inputs)
        x = tf.keras.layers.Bidirectional(
            layer=tf.keras.layers.LSTM(units=16, go_backwards=False),
            backward_layer=tf.keras.layers.GRU(units=16, go_backwards=True),
            merge_mode='concat',
        )(x)
        # no ReLu activation because return can below 0
        outputs = tf.keras.layers.Dense(units=output_units)(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
            

if __name__ == '__main__':

    dataset = dataset.StockPrice(
        symbol='AAPL', 
        n_days_backward=100, 
        n_days_forward=5, 
        batch_size=64
    )
    train_set, validation_set, test_set = dataset.generate(
        drop_outliers=True, 
        split_ratios=(0.8, 0.1, 0.1)
    )
    train_set_for_deployment, _, _ = dataset.generate(
        drop_outliers=True, 
        split_ratios=(1.,0.,0.)
    )

    # # Naive
    # naive = NaiveModel()
    # naive.fit(
    #     train_set=train_set, 
    #     validation_set=validation_set, 
    #     max_epochs=1,
    #     learning_rate=1e-3,
    # )
    # print(naive.evaluate(test_set=test_set))    # [0.00034450148814357817, 0.013859549537301064]

    # # MLP
    # mlp = MLP()
    # mlp.fit(
    #     train_set=train_set, 
    #     validation_set=validation_set, 
    #     max_epochs=256,
    #     learning_rate=1e-3,
    # )
    # print(mlp.evaluate(test_set=test_set))

    # mlp.fit(
    #     train_set=train_set_for_deployment, 
    #     validation_set=None, 
    #     max_epochs=5,
    #     learning_rate=1e-3,
    # ) # only continue training on 5 more epochs
    # print(mlp.predict(data=dataset.last_sample))
    
    # # Conv1D
    # conv1d = Conv1D()
    # conv1d.fit(
    #     train_set=train_set, 
    #     validation_set=validation_set, 
    #     max_epochs=256,
    #     learning_rate=1e-3,
    # )
    # print(conv1d.evaluate(test_set=test_set))

    # conv1d.fit(
    #     train_set=train_set_for_deployment, 
    #     validation_set=None, 
    #     max_epochs=5,
    #     learning_rate=1e-3,
    # ) # only continue training on 5 more epochs
    # print(conv1d.predict(data=dataset.last_sample))


    # # LSTM
    # lstm = LSTM()
    # lstm.fit(
    #     train_set=train_set, 
    #     validation_set=validation_set, 
    #     max_epochs=256,
    #     learning_rate=1e-3,
    # )
    # print(lstm.evaluate(test_set=test_set))

    # lstm.fit(
    #     train_set=train_set_for_deployment, 
    #     validation_set=None, 
    #     max_epochs=5,
    #     learning_rate=1e-3,
    # ) # only continue training on 5 more epochs
    # print(lstm.predict(data=dataset.last_sample))


    # # GRU
    # gru = GRU()
    # gru.fit(
    #     train_set=train_set, 
    #     validation_set=validation_set, 
    #     max_epochs=256,
    #     learning_rate=1e-3,
    # )
    # print(gru.evaluate(test_set=test_set))

    # gru.fit(
    #     train_set=train_set_for_deployment, 
    #     validation_set=None, 
    #     max_epochs=5,
    #     learning_rate=1e-3,
    # ) # only continue training on 5 more epochs
    # print(gru.predict(data=dataset.last_sample))


    # # Bidirectional
    bidirectional = Bidirectional()
    # bidirectional.fit(
    #     train_set=train_set, 
    #     validation_set=validation_set, 
    #     max_epochs=256,
    #     learning_rate=1e-3,
    # )
    # print(bidirectional.evaluate(test_set=test_set))    # [0.00016520229110028595, 0.009174603968858719]

    bidirectional.fit(
        train_set=train_set_for_deployment, 
        validation_set=None, 
        max_epochs=5,
        learning_rate=1e-3,
    ) # only continue training on 5 more epochs
    print(bidirectional.predict(data=dataset.last_sample))






