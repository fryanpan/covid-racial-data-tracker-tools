from typing import List
import os
import datadotworld as dw
import pandas as pd
import numpy as np
import scipy.special

from shared import RaceEthnicity, save_to_dw

def series_to_int(series):
    return pd.to_numeric(series.str.replace(',','').str.replace('N/A','')).astype('Int64')

CASES = 'Cases'
DEATHS = 'Deaths'
HOSP = 'Hosp'
METRICS = [CASES, DEATHS, HOSP]
BASELINES=['White', 'Non-Group', 'Total']

ETHNICITY_PREFIX = 'Ethnicity_'

RACES = [
    RaceEthnicity.TOTAL.value,
    RaceEthnicity.WHITE.value,
    RaceEthnicity.BLACK.value,
    RaceEthnicity.LATINX.value,
    RaceEthnicity.ASIAN.value,
    RaceEthnicity.AIAN.value,
    RaceEthnicity.NHPI.value,
    RaceEthnicity.MULTIRACIAL.value,
    RaceEthnicity.OTHER.value,
    RaceEthnicity.UNKNOWN.value,
    f'{ETHNICITY_PREFIX}{RaceEthnicity.HISPANIC.value}', 
    f'{ETHNICITY_PREFIX}{RaceEthnicity.NON_HISPANIC.value}', 
    f'{ETHNICITY_PREFIX}{RaceEthnicity.UNKNOWN.value}'
]

AGE_GROUP_DATASET_ALL = 'All'
AGE_GROUP_DATA_SET_AGE_GROUP = 'Age Group'

COL_AGE_GROUP_DATASET = 'Age_Group_Dataset'
COL_DATASET = 'Dataset'
COL_DATE = 'Date'
COL_POPULATION = 'Population'
COL_RACE_ETHNICITY = 'Race / Ethnicity'
COL_STATE = 'State'

all_metrics = METRICS[:]

def unpivot(df, initial_columns: List[str], age_group_dataset: str):
    print(df.columns)
    # Unpivot the data to one row per per race / ethnicity, per state, per date
    data = []
    for race in RACES:
        dataset = 'Ethnicity' if race.startswith('Ethnicity') else 'Race'
        race_df = df[initial_columns]

        source_race = 'Latinx' if race == RaceEthnicity.LATINX.value else race
        output_race = race.replace(ETHNICITY_PREFIX, '')
        if output_race == RaceEthnicity.UNKNOWN.value:
            output_race = f'Unknown {dataset}'

        race_df[COL_RACE_ETHNICITY] = output_race
        race_df[COL_DATASET] = dataset
        for metric in METRICS:
            col_name = f'{metric}_{source_race}'
            race_df[metric] = series_to_int(df[col_name])
        race_df[COL_AGE_GROUP_DATASET] = age_group_dataset
        data.append(race_df)

    return pd.concat(data, ignore_index=True)

def sameIndex(df, period):
    return (df[COL_DATASET] == df[COL_DATASET].shift(period)) & (df[COL_STATE] == df[COL_STATE].shift(period)) & (df[COL_RACE_ETHNICITY] == df[COL_RACE_ETHNICITY].shift(period)) 

def compute_deltas(df):
    global all_metrics

    for m in METRICS:
        # Deltas aren't super-meaningful except for checking data quality
        # Since scrapes are 3 and 4 days apart, adjacent deltas are not comparable time periods
        # (and there are also weekly effects)
        # delta = f'{m} Delta'
        # df[delta] = (df[m] - df[m].shift(1)).where(sameIndex(df, 1))

        delta14 = f'{m} Delta 14d'    
        df[delta14] = (df[m] - df[m].shift(4)).where(sameIndex(df, 4))
        all_metrics += [delta14]

def join_population(df):
    # Load population data
    population_query = dw.query(
        'fryanpan13/covid-tracking-racial-data', 
        'SELECT * FROM population_data')
    population_df = population_query.dataframe

    population_df = population_df.rename(columns={'state': COL_STATE, 'state_name': 'State Name', 'race': COL_RACE_ETHNICITY, 
        'dataset': 'Dataset', 'geo_state_name': 'Geo State Name', 
        'population': COL_POPULATION, 
        'population_35': 'Population 35+',
        'population_55': 'Population 55+' })

    population_df[COL_POPULATION] = population_df[COL_POPULATION].astype('Int64')
    population_df['Population 35+'] = population_df['Population 35+'].astype('float').astype('Int64')
    population_df['Population 55+'] = population_df['Population 55+'].astype('float').astype('Int64')


    # Load region data
    region_query = dw.query(
        'fryanpan13/covid-tracking-racial-data', 
        'SELECT * FROM regions')
    region_df  = region_query.dataframe

    region_df = region_df.rename(columns={'state': COL_STATE, 'region': 'Region', 'subregion': 'Subregion'}) \
                         .drop(columns=['state_name'])

    # Load expected deaths data
    expected_deaths_query = dw.query(
        'fryanpan13/covid-tracking-racial-data', 
        'SELECT * FROM expected_deaths')
    expected_deaths_df  = expected_deaths_query.dataframe[['state', 'race', 'dataset', 'expected_deaths']]
    expected_deaths_df = expected_deaths_df.rename(columns={'state': COL_STATE, 
                                                            'dataset': COL_DATASET,
                                                            'race': COL_RACE_ETHNICITY, 
                                                            'expected_deaths': 'Expected Deaths'})
    expected_deaths_df['Expected Deaths'] = expected_deaths_df['Expected Deaths'].astype('float')

    population_df = population_df.set_index([COL_STATE])
    region_df = region_df.set_index([COL_STATE])
    population_df = population_df.merge(region_df, on=[COL_STATE])

    population_index = [COL_DATASET, COL_STATE, COL_RACE_ETHNICITY]
    population_df = population_df.reset_index(drop=False).set_index(population_index)

    expected_deaths_df = expected_deaths_df.reset_index(drop=False).set_index(population_index)
    population_df = population_df.merge(expected_deaths_df, on=population_index)

    print(population_df.columns)
    print(df.columns)

    all_metrics.append("Population")
    all_metrics.append("Population 35+")
    all_metrics.append("Population 55+")
    all_metrics.append('Expected Deaths')

    # Join population
    return df.join(population_df, population_index)

def compute_baselines(df):
    global all_metrics

    # Compute baseline metrics (vs. White, vs. All) and join in
    df.reset_index(drop=False, inplace=True)

    join_index = [COL_STATE, COL_DATE]

    # Compute white metrics
    white_df = df[(df[COL_RACE_ETHNICITY] == 'White') & (df[COL_DATASET] == 'Race')]
    white_df = white_df.add_prefix('White ')
    white_df = white_df.rename(columns={'White Dataset': 'Dataset', 'White State': COL_STATE, 'White Date': COL_DATE })
    white_df = white_df.set_index(join_index)
    white_metrics = ['White ' + m for m in all_metrics]

    # Compute total metrics
    total_df = df[df[COL_RACE_ETHNICITY] == 'Total']
    total_df = total_df.add_prefix('Total ')
    total_df = total_df.rename(columns={'Total Dataset': COL_DATASET, 'Total State': COL_STATE, 'Total Date': COL_DATE })
    total_df = total_df.set_index(join_index)
    total_metrics = ['Total ' + m for m in all_metrics]

    # Compute Unknown metrics
    unknown_join_index = [COL_STATE, COL_DATE, COL_DATASET]
    unknown_df = df[(df[COL_RACE_ETHNICITY] == 'Unknown Race') | (df[COL_RACE_ETHNICITY] == 'Unknown Ethnicity')]
    unknown_df = unknown_df.add_prefix('Unknown ')
    unknown_df = unknown_df.rename(columns={'Unknown Dataset': COL_DATASET, 'Unknown State': COL_STATE, 'Unknown Date': COL_DATE })
    unknown_df = unknown_df.set_index(unknown_join_index)
    unknown_metrics = ['Unknown ' + m for m in all_metrics]
    
    
    df = df.set_index(join_index)
    df = pd.merge(df, white_df[white_metrics], on=join_index, how='left')
    df = pd.merge(df, total_df[total_metrics], on=join_index, how='left')

    df.reset_index(drop=False, inplace=True)
    df.set_index(unknown_join_index)
    return pd.merge(df, unknown_df[unknown_metrics], on=unknown_join_index, how='left')

def compute_per_capita_metrics(df):
    global all_metrics

    # Calculate Poisson Distribution Confidence Intervals
    # https://newton.cx/~peter/2012/06/poisson-distribution-confidence-intervals/
    # This gives the same answers as  https://www.cdc.gov/nchs/data/nvsr/nvsr68/nvsr68_09-508.pdf  (pp. 73-75)

    def confidence_interval_lo(n):
        a = 0.05
        return scipy.special.gammaincinv(n, 0.5 * a)

    def confidence_interval_hi(n):
        a = 0.05
        return scipy.special.gammaincinv(n + 1, 1 - 0.5 * a)

    # Compute per-capita metrics per group
    print("Computing per capita metrics")
    per_capita_metrics = []
    per_capita_suffix = ' per 100,000'

    # No point calculating population per population :P
    # Also the "Delta" metrics from one date to the previous date aren't consistently comparable
    # time periods.  Mainly useful for for debugging data issues
    metrics_to_skip = [COL_POPULATION, 'Population 35+', 'Population 55+', 
                       'Cases Delta', 'Deaths Delta', 'Negatives Delta']
    for source_metric in all_metrics:
        if source_metric in metrics_to_skip: 
            continue

        per_capita_metrics.append(f'{source_metric}{per_capita_suffix}')

        # for group in ['', 'White ', 'Total ', 'Non-Group ']:
        groups = ['', 'White ', 'Total ', 'Non-Group ']
        populations = [COL_POPULATION]
        suffix = ['']
        if source_metric in ['Deaths', 'Deaths Delta 14d']:
            populations.append('Population 35+')
            suffix.append(' 35+')
            per_capita_metrics.append(f'{source_metric}{per_capita_suffix} 35+')

            populations.append('Population 55+')
            suffix.append(' 55+')
            per_capita_metrics.append(f'{source_metric}{per_capita_suffix} 55+')


        for population_field, population_suffix in zip(populations, suffix):
            for group in groups:
                metric_name = group + source_metric
                population = f'{group}{population_field}'
                dest_metric = f'{metric_name}{per_capita_suffix}{population_suffix}'

                print(f'{dest_metric} = {source_metric} / {population} * 100,000')

                source_lo = f'{source_metric} CI Lo'
                source_hi = f'{source_metric} CI Hi'
                source_range = f'{source_metric} CI Range'
                dest_lo = f'{dest_metric} CI Lo'
                dest_hi = f'{dest_metric} CI Hi'
                dest_range = f'{dest_metric} CI Range'

                df[dest_metric] = df[metric_name] / df[population] * 100000
                df[source_lo] = df[metric_name].apply(confidence_interval_lo)
                df[source_hi] = df[metric_name].apply(confidence_interval_hi)
                df[source_range] = df[source_hi] - df[source_lo]

                print(f'{source_metric}, {source_lo}, {source_hi}')
                df[dest_lo] = df[source_lo] / df[population] * 100000
                df[dest_hi] = df[source_hi] / df[population] * 100000
                df[dest_range] = df[dest_hi] - df[dest_lo]
        
    print("Computed per-capita metrics:")
    print('\n'.join(per_capita_metrics))

    all_metrics += per_capita_metrics

    compute_disparity(df, per_capita_metrics)

def compute_disparity(df, per_capita_metrics):
     # TODO: Compute disparity metrics vs. each baseline and non-group 
    print("Computing disparity metrics")
    disparity_metrics = []
    for metric in per_capita_metrics:
        for baseline in BASELINES:
            baseline_metric = f'{baseline} {metric}'
            disparity_metric = f'{metric} Disparity vs. {baseline}'
            disparity_significant = f'{disparity_metric} is Significant'

            print(f'{disparity_metric} = {metric} / {baseline_metric}')
            print(f'{disparity_significant} = {baseline_metric} CI doesn\'t overlap {metric} CI')

            metric_lo = f'{metric} CI Lo'
            metric_hi = f'{metric} CI Hi'
            baseline_lo = f'{baseline_metric} CI Lo'
            baseline_hi = f'{baseline_metric} CI Hi'

            df[disparity_metric] = df[metric] / df[baseline_metric]
            df[disparity_significant] = (df[metric_lo] > df[baseline_hi]).all() or (df[metric_hi] < df[baseline_lo]).all() 

def compute_unknown_non_group(df: pd.DataFrame):
    # Compute non-group metrics
    for m in all_metrics:
        non_group_metric = "Non-Group " + m
        total_metric = "Total " + m
        df[non_group_metric] = df[total_metric] - df[m]

    # Compute unknown %
    for m in all_metrics:
        unknown_metric = f'Unknown {m}'
        total_metric = f'Total {m}'
        df[f'Percent Unknown {m}'] = df[unknown_metric] / df[total_metric]


def doit():
    CRDT_SOURCE_URL='https://docs.google.com/spreadsheets/d/e/2PACX-1vS8SzaERcKJOD_EzrtCDK1dX1zkoMochlA9iHoHg_RSw3V8bkpfk1mpw4pfL5RdtSOyx_oScsUtyXyk/pub?gid=43720681&single=true&output=csv'
    df = pd.read_csv(CRDT_SOURCE_URL, na_filter=False, skipinitialspace=True)

    # Reformat date column
    df = df[df[COL_DATE] != '']
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], format='%Y%m%d')

    df = unpivot(df, [COL_DATE, COL_STATE], AGE_GROUP_DATASET_ALL)    

    output_index = [COL_DATASET, COL_AGE_GROUP_DATASET, COL_STATE, COL_RACE_ETHNICITY, COL_DATE]

    # Sort so all data for each state & ethnicity is adjacent and increasing in date
    df = df.sort_values(output_index)

    # compute differences over time, within the same dataset, state, and race / ethnicity
    compute_deltas(df)

    df = df.set_index(output_index)

    # Join population
    df = join_population(df)

    # Compute baseline metrics (e.g. for White, Total, Unknown)
    df = compute_baselines(df)    

    # Compute non-group and unknown metrics
    compute_unknown_non_group(df)

    df.reset_index(drop=False, inplace=True)

    compute_per_capita_metrics(df)
   
    # Sometimes, if there's no non-group, for example, the denominator is zero.  Clean this up.
    df = df.replace([np.inf, -np.inf], np.nan)

    # save_to_dw(df, 'crdt_output.csv')

    print(df.columns)

    # also save a basic output, without computed metrics
    basic_columns = ['Region', COL_STATE, COL_DATE, COL_DATASET, COL_RACE_ETHNICITY, COL_POPULATION] + METRICS
    basic_df = df[basic_columns]
    # save_to_dw(basic_df, 'crdt_basic_output.csv')




if __name__ == '__main__':
    doit()
