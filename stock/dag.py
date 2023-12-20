import datetime as dt
import pandas as pd
from airflow import DAG
from airflow.decorators import task, dag
import pendulum

import etl

"""
INSTALLATION

export PYTHONPATH=~/katz/machine_learning/project2
export AIRFLOW_HOME=~/katz/machine_learning/project2/airflow

source .venv/bin/activate
AIRFLOW_VERSION=2.7.3
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip3 install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

"""

@task()
def update_sp500_list():
    company = etl.Company()
    company.extract().transform().load()

@task()
def update_stock_prices():
    # get all symbols from S&P 500
    symbols = pd.read_sql(
        """
        select distinct "symbol"
        from "company";
        """,
        con=etl.engine,
    ).squeeze()
    # get data each of symbols
    daily = etl.Daily()
    for symbol in symbols:
        if pd.read_sql(
            f"""
            select "symbol" from "daily"
            where "symbol" = '{symbol}'
            """,
            con=etl.engine,
        ).empty:
            size = 'full'
        else:
            size = 'compact'
        daily.extract(symbol=symbol, size=size).transform().load()    


@dag(
    schedule='0 18 * * *', # At 6:00 PM every day
    start_date=pendulum.datetime(2023,12,12,tz="EST"),
    catchup=False,
)
def etl_pipeline():
    update_sp500_list()
    update_stock_prices()

etl_pipeline()


