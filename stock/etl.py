import re
import abc
import time
import typing
import json
import bs4
import requests
import sqlalchemy
import pandas as pd

with open('APIKEY') as file:
    APIKEY = file.read()

with open('DBAUTH') as file:
    DBAUTH = file.read()

engine = sqlalchemy.create_engine(DBAUTH)


class BaseETL(abc.ABC):

    """
    Base class of all ETL processes
    """

    def __init__(self):
        # Placeholder for the extracted data
        self.__data: typing.Optional[pd.DataFrame] = None

    #read-only
    @property
    def data(self) -> pd.DataFrame:
        """
        Returns the extracted and transformed data.

        Returns:
        -------
        pd.DataFrame
            The extracted data as a DataFrame.
        """
        return self.__data

    @abc.abstractmethod
    def extract(self) -> typing.Self:
        pass

    @abc.abstractmethod
    def transform(self) -> typing.Self:
        pass

    @abc.abstractmethod
    def load(self) -> None:
        pass


class Company(BaseETL):

    """
    A class for extracting, transforming, and loading data about companies 
    listed in the S&P 500.

    This class scrapes company data from Wikipedia, processes it, and then 
    loads it into the database.
    """

    URL = r'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    def extract(self) -> typing.Self:
        """
        Extracts company data from the Wikipedia page.

        Scrapes the table of companies listed in the S&P 500 from Wikipedia and 
        stores it in a DataFrame.

        Returns:
        -------
        Company
            The instance itself, allowing for method chaining.
        """

        # Make a request to the Wikipedia page
        r = requests.get(self.URL)
        # Parse the HTML content
        soup = bs4.BeautifulSoup(markup=r.text, features='lxml')
        # Find the table body
        tbody: bs4.element.Tag = soup.find(name='tbody')
        # Extract rows from the table, skipping the header
        rows: typing.List[bs4.element.Tag] = tbody.find_all(name='tr')[1:]
        records: typing.List[typing.Tuple[str, str, str]] = []

        # Iterate over each row and extract the data
        for row in rows:
            strings = row.stripped_strings
            symbol, name, gics = strings.__next__(), strings.__next__(), strings.__next__()
            records.append((symbol, name, gics))
        
        # Convert the list of records into a DataFrame
        self.__data = pd.DataFrame(data=records, columns=['symbol', 'name', 'gics'])
        return self

    def transform(self) -> typing.Self:
        """
        Transforms the extracted data. Applies cleaning operations to the data.

        Returns:
        -------
        Company
            The instance itself, allowing for method chaining.
        """

        # Clean the text data in the DataFrame
        self.__data = self.__data.map(self.__clean_text)
        return self

    def load(self) -> None:
        """
        Loads the transformed data into a database, replacing any existing records.
        """

        # List of symbols to be inserted
        inserting_symbols = self.__data['symbol'].to_list()
        # Define the database table
        company = sqlalchemy.Table(
            'company', 
            sqlalchemy.MetaData(), 
            autoload_with=engine
        )
        # Establish a connection to the database
        with engine.begin() as db_connection:
            # Delete existing records for these symbols
            db_connection.execute(
                sqlalchemy.delete(company).where(
                    company.columns.symbol.in_(inserting_symbols)
                )
            )
            # Insert new records
            db_connection.execute(
                statement=sqlalchemy.insert(company), 
                parameters=self.__data.to_dict(orient='records'),
            )

    @staticmethod
    def __clean_text(raw_text: str) -> str:
        """
        Cleans a given text string.

        Removes extra spaces and strips leading/trailing whitespace.

        Parameters:
        ----------
        raw_text : str
            The raw text to be cleaned.

        Returns:
        -------
        str
            The cleaned text.
        """

        return re.sub(r'\s{2,}', ' ', raw_text.strip())


class Daily(BaseETL):

    """
    A class for extracting, transforming, and loading daily stock data from Alpha Vantage API.
    """

    URL = r'https://www.alphavantage.co/query'

    def extract(
        self, 
        symbol: str, 
        size: typing.Literal['full','compact'] = 'compact'
    ) -> typing.Self:
        """
        Extracts stock data from the Alpha Vantage API.

        Parameters:
        ----------
        symbol : str
            The stock symbol to fetch data for.
        size : typing.Literal['full', 'compact'], optional
            The size of the dataset to fetch ('full' or 'compact'). Default is 'compact'.

        Returns:
        -------
        typing.Self
            The instance itself, allowing for method chaining.
        """

        print(symbol)
        column_names = ['date','symbol','open','high','low','close','volume']
        while True:
            r = requests.get(
                self.URL,
                params={
                    'function'      : 'TIME_SERIES_DAILY_ADJUSTED',
                    'datatype'      : 'json',
                    'symbol'        : symbol,
                    'outputsize'    : size,
                    'apikey'        : APIKEY,
                    'entitlement'   : 'delayed',
                }
            )
            response: typing.Optional[dict[str, dict]] = r.json()
            # hitting limit number of requests per minute
            if 'Information' in response.keys():
                # sleep and retry
                sleep_in_second = 1
                print(f'Request limit hit, wait for {sleep_in_second}s.')
                time.sleep(sleep_in_second)
            # stock symbol not found in the API
            elif 'Error Message' in response.keys():
                print(f'{symbol} not found')
                # create an empty dataframe to disable the effect of transform() and load()
                self.__data = pd.DataFrame(
                    columns=column_names
                )
                return self
            # succeed
            else:
                break
                
        records = []
        for date, record in response.get('Time Series (Daily)').items():
            open_price  = float(record.get('1. open'))
            high_price  = float(record.get('2. high'))
            low_price   = float(record.get('3. low'))
            close_price = float(record.get('4. close'))
            volume      = float(record.get('6. volume'))
            records.append(
                (date, symbol, open_price, high_price, low_price, close_price, volume)
            )

        self.__data = pd.DataFrame(data=records, columns=column_names)
        return self

    def transform(self) -> typing.Self:
        """
        Transforms the extracted data.

        Returns:
        -------
        typing.Self
            The instance itself, allowing for method chaining.
        """

        # Apply transformation (strip) to the 'date' column
        self.__data['date'] = self.__data['date'].map(lambda x: x.strip())
        return self

    def load(self) -> None:
        """
        Loads the transformed data into the database, replacing any existing records.
        """

        if self.__data.empty:
            return

        daily = sqlalchemy.Table(
            'daily', 
            sqlalchemy.MetaData(), 
            autoload_with=engine
        )
        with engine.begin() as db_connection:
            inserting_dates = self.__data['date'].to_list()
            inserting_symbol = set(self.__data['symbol']).pop()
            # Delete existing records for the same dates and symbol
            db_connection.execute(
                statement=sqlalchemy.delete(daily).where(
                    daily.columns.symbol == inserting_symbol,
                    daily.columns.date.in_(inserting_dates),
                )
            )
            # Insert new records
            db_connection.execute(
                statement=sqlalchemy.insert(daily),
                parameters=self.__data.to_dict(orient='records'),
            )


if __name__ == '__main__':

    # update the S&P 500 list
    company = Company()
    company.extract().transform().load()
    # get all symbols from S&P 500
    symbols = pd.read_sql(
        """
        select distinct "symbol"
        from "company";
        """,
        con=engine,
    ).squeeze().to_list()

    # get data each of symbols
    daily = Daily()
    for symbol in symbols:
        if pd.read_sql(
            f"""
            select "symbol" from "daily"
            where "symbol" = '{symbol}'
            """,
            con=engine,
        ).empty:
            size = 'full'
        else:
            size = 'compact'
        daily.extract(symbol=symbol, size=size).transform().load()

