import pandas as pd
import model
import dataset

def predict(
    symbol: str,
    backward: int,
    forward: int,
):

    stock_price = dataset.StockPrice(
        symbol=symbol, 
        n_days_backward=backward, 
        n_days_forward=forward, 
        batch_size=64
    )
    train_set_for_deployment, _, _ = stock_price.generate(
        drop_extremes=True, 
        split_ratios=(1.,0.,0.)
    )
    bidirectional = model.Bidirectional()
    bidirectional.fit(
        train_set=train_set_for_deployment, 
        validation_set=None, 
        max_epochs=5,
        learning_rate=1e-3,
    )
    output: pd.DataFrame = bidirectional.predict(data=stock_price.last_sample)
    output['price'] = output['price'].round(decimals=2)
    output['date'] = output['date'].dt.strftime('%Y-%m-%d')
    output = output.set_index('date')
    actual = output.loc[output['type']=='actual', 'price']
    predicted = output.loc[output['type']=='predicted', 'price']

    return {
        'symbol':       symbol,
        'actual':       actual.to_dict(),
        'predicted':    predicted.to_dict()
    }
    


