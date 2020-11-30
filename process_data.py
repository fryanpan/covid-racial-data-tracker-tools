import datadotworld as dw
import pandas as pd
import numpy as np
import scipy.special

def series_to_int(series):
    return pd.to_numeric(series.str.replace(',','').str.replace('N/A','')).astype('Int64')

def doit():    
    CRDT_SOURCE_URL='https://docs.google.com/spreadsheets/d/e/2PACX-1vS8SzaERcKJOD_EzrtCDK1dX1zkoMochlA9iHoHg_RSw3V8bkpfk1mpw4pfL5RdtSOyx_oScsUtyXyk/pub?gid=43720681&single=true&output=csv'

    # Define some constants that match the source columns
    CASES = 'Cases'
    DEATHS = 'Deaths'
    HOSP = 'Hosp'
    METRICS = [CASES, DEATHS, HOSP]

    RACES = ['Total', 'White', 'Black', 'LatinX', 'Asian', 
    'AIAN', 'NHPI', 'Multiracial', 'Other', 'Unknown',
    'Ethnicity_Hispanic', 'Ethnicity_NonHispanic', 'Ethnicity_Unknown']

    # Baseline race for comparison
    BASELINE = 'White'

    df = pd.read_csv(CRDT_SOURCE_URL, na_filter=False, skipinitialspace=True)

    population_query = dw.query(
        'fryanpan13/covid-tracking-racial-data', 
        'SELECT * FROM population_data')
    population_df = population_query.dataframe

    population_df = population_df.rename(columns={'state': 'State', 'state_name': 'State Name', 'race': 'Race / Ethnicity', 'dataset': 'Dataset', 'geo_state_name': 'Geo State Name', 'population': 'Population' })

    population_df['Population'] = population_df['Population'].astype('Int64')

    population_index = ['Dataset', 'State', 'Race / Ethnicity']
    population_df = population_df.set_index(population_index)

    print(df.columns)

    # Reformat date column
    df = df[df['Date'] != '']
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

    # Unpivot the data to one row per per race / ethnicity, per state, per date


    data = []
    for race in RACES:
        race_df = df[['Date', 'State']]
        race_df['Race / Ethnicity'] = race.replace('Ethnicity_', '')

        dataset = 'Ethnicity' if race.startswith('Ethnicity') else 'Race'
        race_df['Dataset'] = dataset
        for metric in METRICS:
            col_name = f'{metric}_{race}'
            race_df[metric] = series_to_int(df[col_name])
        data.append(race_df)

    df = pd.concat(data, ignore_index=True)


    # Sort so all data for each state & ethnicity is adjacent and increasing in date
    df = df.sort_values(['Dataset', 'State', 'Race / Ethnicity', 'Date'])

    def sameIndex(df, period):
        return (df['Dataset'] == df['Dataset'].shift(period)) & (df['State'] == df['State'].shift(period)) & (df['Race / Ethnicity'] == df['Race / Ethnicity'].shift(period)) 

    # compute differences over time, within the same dataset, state, and race / ethnicity
    all_metrics = METRICS[:]
    sub_metrics = ['', ' Delta', ' Delta 14d']
    for m in METRICS:
        delta = f'{m} Delta'
        delta14 = f'{m} Delta 14d'    
        df[delta] = (df[m] - df[m].shift(1)).where(sameIndex(df, 1))
        df[delta14] = (df[m] - df[m].shift(4)).where(sameIndex(df, 4))
        all_metrics += [delta, delta14]

    output_index = ['Dataset', 'State', 'Race / Ethnicity', 'Date']
    df = df.set_index(output_index)

    # Join population
    df = df.join(population_df, population_index)

    all_metrics.append("Population")

    # Compute baseline metrics (vs. White, vs. All) and join in
    df.reset_index(drop=False, inplace=True)

    white_df = df[(df['Race / Ethnicity'] == BASELINE) & (df['Dataset'] == 'Race')]
    white_df = white_df.add_prefix('White ')
    white_df = white_df.rename(columns={'White Dataset': 'Dataset', 'White State': 'State', 'White Date': 'Date' })
    white_metrics = ['White ' + m for m in all_metrics]

    total_df = df[df['Race / Ethnicity'] == 'Total']
    total_df = total_df.add_prefix('Total ')
    total_df = total_df.rename(columns={'Total Dataset': 'Dataset', 'Total State': 'State', 'Total Date': 'Date' })
    total_metrics = ['Total ' + m for m in all_metrics]

    join_index = ['State', 'Date']
    df = df.set_index(join_index)
    white_df = white_df.set_index(join_index)
    total_df = total_df.set_index(join_index)

    df = pd.merge(df, white_df[white_metrics], on=join_index, how='left')
    df = pd.merge(df, total_df[total_metrics], on=join_index, how='left')

    df.reset_index(drop=False, inplace=True)

    # Compute non-group metrics
    for m in all_metrics:
        non_group_metric = "Non-Group " + m
        total_metric = "Total " + m
        df[non_group_metric] = df[total_metric] - df[m]

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
    metrics_to_skip = ['Population', 'Cases Delta', 'Deaths Delta', 'Negatives Delta']
    for source_metric in all_metrics:
        if source_metric in metrics_to_skip: 
            continue

        per_capita_metrics.append(f'{source_metric}{per_capita_suffix}')

        for group in ['', 'White ', 'Total ', 'Non-Group ']:
            metric_name = group + source_metric
            population = group + 'Population'
            dest_metric = f'{metric_name}{per_capita_suffix}'

            print(f'{dest_metric} = {source_metric} / {population} * 100,000')

            source_lo = f'{source_metric} CI Lo'
            source_hi = f'{source_metric} CI Hi'
            dest_lo = f'{dest_metric} CI Lo'
            dest_hi = f'{dest_metric} CI Hi'

            df[dest_metric] = df[metric_name] / df[population] * 100000
            df[source_lo] = df[metric_name].apply(confidence_interval_lo)
            df[source_hi] = df[metric_name].apply(confidence_interval_hi)

            print(f'{source_metric}, {source_lo}, {source_hi}')
            print(df[source_lo])
            df[dest_lo] = df[source_lo] / df[population] * 100000
            df[dest_hi] = df[source_hi] / df[population] * 100000        

    all_metrics += per_capita_metrics
    print("")


    # TODO: Compute disparity metrics vs. each baseline and non-group 
    print("Computing disparity metrics")
    disparity_metrics = []
    for metric in per_capita_metrics:
        for baseline in ['White', 'Non-Group']:
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

    print("")

    save_to_dw(df, 'crdt_output.csv')

    print(df.columns)
    # also save a basic output, without computed metrics
    basic_columns = ['State', 'Date', 'Dataset', 'Race / Ethnicity', 'Population'] + METRICS
    basic_df = df[basic_columns]
    save_to_dw(basic_df, 'crdt_basic_output.csv')


def save_to_dw(df, filename):
    df.to_csv(f'/tmp/{filename}', index=True)
    client = dw.api_client()
    client.upload_files('fryanpan13/covid-tracking-racial-data',files=f'/tmp/{filename}')


if __name__ == '__main__':
    doit()
