import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
import model, dataset


def univariate_numerical(data: pd.Series):
    
    """
    This function performs univariate analysis on a numerical variable using seaborn
    It generates a boxplot and a histogram with a Kernel Density Estimate (KDE).

    Parameters:
    ----------
    - data: pd.Series
        A named Pandas Series object containing numerical data for analysis.

    The function creates a two-subplot figure:
    - The first subplot is a boxplot, which provides information about the median,
    quartiles, and outliers of the distribution.
    - The second subplot is a histogram with a KDE curve that provides
    a smooth estimate of the distribution.
    """

    # Create a new figure for the plots.
    plt.figure()

    # Create the first subplot: a boxplot for the data series.
    plt.subplot(121)  # 1 row, 2 columns, 1st subplot
    sns.boxplot(data, orient='v')  # 'v' for vertical orientation
    plt.title(f'Boxplot of {data.name}')  # Set the title with the series' name
    plt.xlabel('')  # No label for the x-axis

    # Create the second subplot: a histogram with a KDE.
    plt.subplot(122)  # 1 row, 2 columns, 2nd subplot
    # Generate a histogram with KDE; adjust the bandwidth for smoothing.
    ax = sns.histplot(data, bins=20, kde=True, kde_kws={"bw_adjust": 3})
    ax.set_ylabel('')  # No label for the y-axis
    plt.title(f'Histogram of {data.name}')  # Set the title with the series' name

    # Display the figure with both subplots.
    plt.show()


def bivariate_numerical_numerical(data: pd.DataFrame, x: str, y: str):
    """
    Create a bivariate scatter plot using Seaborn to visualize the relationship between two numerical variables.
    
    Parameters:
    ----------
    - data: pd.DataFrame
        The Pandas DataFrame containing the data.
    - x: str
        The name of the column in the DataFrame to be used as the x-axis in the scatter plot.
    - y: str
        The name of the column in the DataFrame to be used as the y-axis in the scatter plot.
    
    The function plots a scatter plot where each point represents an observation with its x and y values.
    """
    # Create a new figure for plotting.
    plt.figure()
    # Plot a scatter plot with the specified columns as the x and y axes.
    # `linewidth=0` removes the outline of the markers.
    sns.scatterplot(x=x, y=y, data=data, linewidth=0)
    # Set the title of the plot. The title indicates the variables being compared.
    plt.title(f'{x} vs. {y}')
    # Display the plot.
    plt.show()


def candlestick(data: pd.DataFrame) -> None:
    """
    Generates a candlestick chart from a given DataFrame.

    This function creates a candlestick chart, commonly used in financial analysis 
    to depict the price movement of securities, derivatives, or currency. 
    It represents the high, low, open, and closing prices of a stock or 
    trading instrument over a specified period.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the stock market data with columns 'Date', 
        'Open', 'High', 'Low', 'Close', and 'Volume'.

    Returns
    -------
    None
        Displays the candlestick chart.
    """

    # Convert 'Date' column to datetime and set as index
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    # Create subplot layout for candlestick and volume plots
    _, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    # Define market colors and style for the plot
    mc = mpf.make_marketcolors(up='green', down='red', edge='black', volume='black', inherit=True)
    s  = mpf.make_mpf_style(marketcolors=mc)
    # Plotting the candlestick chart
    mpf.plot(data, type='candle', ax=axes[0], volume=axes[1], style=s, show_nontrading=False)
    # Adjusting the plot spines for aesthetics
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
    # Keep the x-axis visible on the volume subplot
    axes[1].spines['bottom'].set_visible(True)
    # Display the plot
    plt.show()


def plot_price(data: pd.DataFrame, symbol: str):
    data['date'] = pd.to_datetime(data['date'])
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(data['date'], data['price'], color='blue', label='actual')
    plt.title(f'{symbol}')
    plt.xlabel('date')
    plt.ylabel('price')
    plt.legend()
    plt.show()

def plot_price_with_prediction(data: pd.DataFrame, symbol: str, model_name: str):
    data['date'] = pd.to_datetime(data['date'])
    # Separate the actual and predicted prices
    actual_prices = data.loc[data['type']=='actual']
    predicted_prices = data.loc[data['type']=='predicted']
    predicted_prices = pd.concat([actual_prices.tail(1), predicted_prices], axis=0)
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(actual_prices['date'], actual_prices['price'], color='blue', label='actual')
    plt.plot(predicted_prices['date'], predicted_prices['price'], color='red', linestyle='dashed', label=f'predicted')
    plt.title(f'prediction on {symbol}: {model_name}')
    plt.xlabel('date')
    plt.ylabel('price')
    plt.legend(loc='lower left')
    plt.show()

def plot_return(*returns: pd.Series, symbol: str):
    """
    Plots the return series for a given stock symbol.

    This function takes multiple pandas Series objects representing stock returns
    and plots them in individual subplots for comparison.

    Parameters
    ----------
    *returns : pd.Series
        One or more pandas Series objects containing stock return data.
    symbol : str
        The stock symbol associated with the return data.

    Returns
    -------
    None
        Displays the plot and saves it as 'test.png'.
    """

    # Determine the number of return series to plot
    num_series = len(returns)
    # Create subplots for each return series
    fig, axes = plt.subplots(num_series, 1, figsize=(10, 2*num_series), sharex=True)
    # Plot each return series in a separate subplot
    for i, s in enumerate(returns):
        axes[i].plot(s.index, s.values)
        axes[i].legend([s.name], loc='upper left')
    # Set the title for the entire figure
    fig.suptitle(symbol)
    # Adjust layout for clear visibility of each subplot
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    # Save the plot as an image file
    plt.savefig('test.png')
    # Display the plot
    plt.show()


if __name__ == '__main__':

    dataset = dataset.StockPrice(symbol='AAPL', n_days_backward=100, n_days_forward=5, batch_size=64)
    train_set, validation_set, test_set = dataset.generate(drop_outliers=True, split_ratios=(0.8, 0.1, 0.1))

    train_set_for_deployment, _, _ = dataset.generate(drop_outliers=True, split_ratios=(1.,0.,0.))

    # candlestick(dataset.full_df[['date','open','high','low','close','volume']].tail(50).rename(columns=lambda x: x.title()))
    

    bidirectional = model.Bidirectional()
    bidirectional.fit(
        train_set=train_set, 
        validation_set=validation_set, 
        max_epochs=256,
        learning_rate=1e-3
    )
    print(bidirectional.evaluate(test_set=test_set))

    bidirectional.fit(
        train_set=train_set_for_deployment, 
        validation_set=None, 
        max_epochs=5,
        learning_rate=1e-3
    ) # only continue training on 5 more epochs
    p = bidirectional.predict(data=dataset.last_sample)
    plot_price_with_prediction(
        data=p, 
        symbol=dataset.symbol, 
        model_name=bidirectional.__class__.__name__.lower()
    )

    # data = dataset.full_df
    # price_return_columns = ['open_r','high_r','low_r','close_r']
    # feeding_data = data.dropna(axis=0, how='any')
    # upper_outliers = feeding_data[price_return_columns].quantile(0.999)
    # lower_outliers = feeding_data[price_return_columns].quantile(0.001)
    # for return_column in price_return_columns:
    #     upper_threshold = upper_outliers[return_column]
    #     lower_threshold = lower_outliers[return_column]
    #     feeding_data = feeding_data.loc[
    #         (feeding_data[return_column] < upper_threshold) 
    #         & (feeding_data[return_column] > lower_threshold)
    #     ]

    # feeding_data = feeding_data.tail(252)
    # plot_return(
    #     feeding_data[['date','open_r']].set_index('date').squeeze(),
    #     feeding_data[['date','high_r']].set_index('date').squeeze(),
    #     feeding_data[['date','low_r']].set_index('date').squeeze(),
    #     feeding_data[['date','close_r']].set_index('date').squeeze(),
    #     symbol=stock_price.symbol,
    # )


    # bivariate_numerical_numerical(feeding_data, 'open_r', 'close_r')

    data = dataset.full_df[['date','close']].rename({'close': 'price'}, axis=1).tail(252)
    plot_price(data, dataset.symbol)
