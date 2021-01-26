import os
import requests
import datadotworld as dw
import pandas as pd
import numpy as np
import scipy.special
from shared import RaceEthnicity, save_to_dw

def series_to_int(series):
    return pd.to_numeric(series.str.replace(',','').str.replace('N/A','')).astype('Int64')


SEX_MAP = {
    0: 'Total',
    1: 'Male',
    2: 'Female'
}

ORIGIN_MAP = {
    0: 'Total',
    1: 'Not Hispanic',
    2: 'Hispanic'
}

RACE_MAP = {
    1: RaceEthnicity.WHITE.value,
    2: RaceEthnicity.BLACK.value,
    3: RaceEthnicity.AIAN.value,
    4: RaceEthnicity.ASIAN.value,
    5: RaceEthnicity.NHPI.value,
    6: RaceEthnicity.MULTIRACIAL.value
}

def get_age_ranges():
    # Load population data
    age_range_query = dw.query(
        'fryanpan13/covid-tracking-racial-data', 
        'SELECT * FROM standard_population_weights')
    return age_range_query.dataframe


def generate_output_by_single_age():  
    df = pd.read_csv('sc-est2019-alldata6.csv')
    
    df = df.replace({ 'SEX': SEX_MAP, 'ORIGIN': ORIGIN_MAP, 'RACE': RACE_MAP})
    df = df[['NAME', 'SEX', 'ORIGIN', 'RACE', 'AGE', 'POPESTIMATE2019']]
    df = df[df['SEX'] == 'Total']
    df = df.drop('SEX', axis='columns')


    df = df.rename({'NAME': 'State', 
                    'ORIGIN': 'Ethnicity', 
                    'RACE': 'Race', 
                    'AGE': 'Age',
                    'POPESTIMATE2019': 'Population'}, 
                    axis='columns')
    print(df)

    non_hispanic_df = df[df['Ethnicity'] == 'Not Hispanic'].drop('Ethnicity', axis='columns')
    hispanic_df = df[df['Ethnicity'] == 'Hispanic'].drop('Ethnicity', axis='columns')
    total_df = df[df['Ethnicity'] == 'Total'].drop('Ethnicity', axis='columns')

    non_hispanic_total = non_hispanic_df.groupby(by=['State', 'Age']).sum()
    hispanic_total = hispanic_df.groupby(by=['State', 'Age']).sum()

    df = non_hispanic_df.merge(total_df, how='left', on=['State', 'Race', 'Age']) \
                        .rename({
                                    'Race': 'Race_Ethnicity',
                                    'Population_x': 'Population_NonHispanic',
                                    'Population_y': 'Population_Total'
                                }, axis='columns')

    standard_column_order = ['State', 'Age', 'Race_Ethnicity', 'Population_NonHispanic', 'Population_Total']

    # Generate LatinX and Hispanic from totals
    hispanic_total['Race_Ethnicity'] = RaceEthnicity.HISPANIC.value
    hispanic_total['Population_NonHispanic'] = 0
    hispanic_total.rename({'Population': 'Population_Total'}, axis='columns', inplace=True)
    hispanic_total = hispanic_total.reset_index()[standard_column_order]
    df = df.append(hispanic_total)
    
    hispanic_total['Race_Ethnicity'] = RaceEthnicity.LATINX.value
    df = df.append(hispanic_total)

    # Generate NonHispanic from totals
    non_hispanic_total['Race_Ethnicity'] = RaceEthnicity.NON_HISPANIC.value
    non_hispanic_total['Population_Total'] = non_hispanic_total['Population']
    non_hispanic_total.rename({'Population': 'Population_NonHispanic'}, axis='columns', inplace=True)
    non_hispanic_total = non_hispanic_total.reset_index()[standard_column_order]
    df = df.append(non_hispanic_total)
    df = df[standard_column_order]
    
    print('Single Age Output')
    print(df)
    return df

def generate_output_by_age_range(single_age_df: pd.DataFrame):
    age_ranges_df = get_age_ranges()
    print(age_ranges_df)

    single_age_df['key'] = 0
    age_ranges_df['key'] = 0

    df = single_age_df.merge(age_ranges_df, how='left', on='key')
    df.drop('key', 'columns', inplace=True)
    df = df.reset_index()
    print(df)

    df['in_range'] = (df['Age'] >= df['min_age']) & (df['Age'] <= df['max_age'])
    df = df[df['in_range']]
    print(df)

    df = df.reset_index()
    df = df.groupby(['State', 'Race_Ethnicity', 'age_range', 'min_age', 'max_age']) \
           .agg({
                'Population_NonHispanic': 'sum',
                'Population_Total': 'sum'
            }) \
           .reset_index() \
           .rename({'age_range': 'Age Range', 'min_age': 'Min Age', 'max_age': 'Max Age'}, axis='columns')
    print(df)
    return df

def doit():    
    single_age_df = generate_output_by_single_age()
    save_to_dw(single_age_df, 'population_estimates_single_age.csv')

    age_range_df = generate_output_by_age_range(single_age_df)
    save_to_dw(age_range_df, 'population_estimates_age_ranges.csv')
    

if __name__ == '__main__':
    doit()
