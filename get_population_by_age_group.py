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


COL_AGE = 'Age'
COL_DATASET = 'Dataset' # used to match CRDT Dataset names ('Race' and 'Ethnicity')
COL_POPULATION = 'Population'
COL_POPULATION_NON_HISPANIC = 'Population NonHispanic'
COL_POPULATION_TOTAL = 'Population Total'
COL_RACE_ETHNICITY = 'Race / Ethnicity'
COL_RACE_INCLUDES_HISPANIC = 'Race Includes Hispanic'
COL_STATE = 'State'
COL_STATE_NAME = 'State Name'
COL_TILEGRAM = 'Tilegram Geo State Name'

def get_age_ranges():
    # Load standard weights by age range 
    age_range_query = dw.query(
        'fryanpan13/covid-tracking-racial-data', 
        'SELECT * FROM standard_population_weights')
    return age_range_query.dataframe

def get_state_reporting_category():
    state_reporting_df = dw.query(
        'fryanpan13/covid-tracking-racial-data', 
        'SELECT * FROM state_reporting_category')
    return state_reporting_df.dataframe


def generate_output_by_single_age():  
    state_reporting_df = get_state_reporting_category() \
        .rename({ 'state': COL_STATE, 
                  'state_name': COL_STATE_NAME,
                  'tilegram_geo_state_name': COL_TILEGRAM,
                  'race_includes_hispanic': COL_RACE_INCLUDES_HISPANIC }, axis=1)
    print(state_reporting_df)
    df = pd.read_csv('sc-est2019-alldata6.csv')
    
    df = df.replace({ 'SEX': SEX_MAP, 'ORIGIN': ORIGIN_MAP, 'RACE': RACE_MAP})
    df = df[['NAME', 'SEX', 'ORIGIN', 'RACE', 'AGE', 'POPESTIMATE2019']]
    df = df[df['SEX'] == 'Total']
    df = df.drop('SEX', axis='columns')


    df = df.rename({'NAME': COL_STATE_NAME, 
                    'ORIGIN': 'Ethnicity', 
                    'RACE': 'Race', 
                    'AGE': COL_AGE,
                    'POPESTIMATE2019': COL_POPULATION}, 
                    axis='columns')
    df = df.merge(state_reporting_df, how='left', on=[COL_STATE_NAME])

    non_hispanic_df = df[df['Ethnicity'] == 'Not Hispanic'].drop('Ethnicity', axis='columns')
    hispanic_df = df[df['Ethnicity'] == 'Hispanic'].drop('Ethnicity', axis='columns')
    total_df = df[df['Ethnicity'] == 'Total'].drop('Ethnicity', axis='columns')

    non_hispanic_total = non_hispanic_df.groupby(by=[COL_STATE, COL_STATE_NAME, COL_TILEGRAM, COL_AGE, COL_RACE_INCLUDES_HISPANIC]).sum()
    non_hispanic_total[COL_DATASET] = 'Ethnicity'

    hispanic_total = hispanic_df.groupby(by=[COL_STATE, COL_STATE_NAME, COL_TILEGRAM, COL_AGE, COL_RACE_INCLUDES_HISPANIC]).sum()
    hispanic_total[COL_DATASET] = 'Ethnicity'

    df = non_hispanic_df.merge(total_df, how='left', on=[COL_STATE, COL_STATE_NAME, COL_TILEGRAM, 'Race', COL_AGE, COL_RACE_INCLUDES_HISPANIC]) \
                        .rename({
                                    'Race': COL_RACE_ETHNICITY,
                                    'Population_x': COL_POPULATION_NON_HISPANIC,
                                    'Population_y': COL_POPULATION_TOTAL
                                }, axis='columns')
    df[COL_DATASET] = 'Race'


    mask = df[COL_RACE_INCLUDES_HISPANIC] == 'no'
    df.loc[mask, COL_POPULATION] = df.loc[mask, COL_POPULATION_NON_HISPANIC]

    mask = df[COL_RACE_INCLUDES_HISPANIC] == 'yes'
    df.loc[mask, COL_POPULATION] = df.loc[mask, COL_POPULATION_TOTAL]

    standard_column_order = [COL_STATE, COL_STATE_NAME, COL_TILEGRAM, COL_AGE, COL_DATASET, COL_RACE_ETHNICITY, COL_RACE_INCLUDES_HISPANIC, COL_POPULATION, COL_POPULATION_NON_HISPANIC, COL_POPULATION_TOTAL]
    df = df[standard_column_order]

    # Generate LatinX and Hispanic from totals
    hispanic_total[COL_RACE_ETHNICITY] = RaceEthnicity.HISPANIC.value
    hispanic_total[COL_POPULATION_NON_HISPANIC] = 0
    hispanic_total[COL_POPULATION_TOTAL] = hispanic_total[COL_POPULATION]
    hispanic_total = hispanic_total.reset_index()[standard_column_order]
    df = df.append(hispanic_total)
    
    hispanic_total[COL_RACE_ETHNICITY] = RaceEthnicity.LATINX.value
    df = df.append(hispanic_total)

    # Generate NonHispanic from totals
    non_hispanic_total[COL_RACE_ETHNICITY] = RaceEthnicity.NON_HISPANIC.value
    non_hispanic_total[COL_POPULATION_TOTAL] = non_hispanic_total[COL_POPULATION]
    non_hispanic_total[COL_POPULATION_NON_HISPANIC] = non_hispanic_total[COL_POPULATION]
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

    df['in_range'] = (df[COL_AGE] >= df['min_age']) & (df[COL_AGE] <= df['max_age'])
    df = df[df['in_range']]
    print(df)

    df = df.reset_index()
    df = df.groupby([COL_STATE, COL_STATE_NAME, COL_TILEGRAM, COL_DATASET, COL_RACE_ETHNICITY, COL_RACE_INCLUDES_HISPANIC, 'age_range', 'min_age', 'max_age', 'standard_weight']) \
           .agg({
                COL_POPULATION: 'sum',
                COL_POPULATION_NON_HISPANIC: 'sum',
                COL_POPULATION_TOTAL: 'sum'
            }) \
           .reset_index() \
           .rename({'age_range': 'Age Range', 'min_age': 'Min Age', 'max_age': 'Max Age'}, axis='columns')
    return df

def doit():    
    single_age_df = generate_output_by_single_age()
    save_to_dw(single_age_df, 'population_estimates_single_age.csv')

    age_range_df = generate_output_by_age_range(single_age_df)
    save_to_dw(age_range_df, 'population_estimates_age_ranges.csv')
    

if __name__ == '__main__':
    doit()
