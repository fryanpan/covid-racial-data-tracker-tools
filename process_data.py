import datadotworld as dw
import pandas as pd
import numpy as np
import scipy.special

def series_to_int(series):
    return pd.to_numeric(series.str.replace(',','').str.replace('N/A','')).astype('Int64')

CRDT_SOURCE_URL='https://docs.google.com/spreadsheets/u/1/d/e/2PACX-1vR_xmYt4ACPDZCDJcY12kCiMiH0ODyx3E1ZvgOHB8ae1tRcjXbs_yWBOA4j4uoCEADVfC1PS2jYO68B/pub?output=csv&gid=902690690'

CASES = 'Cases'
DEATHS = 'Deaths'
NEGATIVES = 'Negatives'
METRICS = [CASES, DEATHS, NEGATIVES]

df = pd.read_csv(CRDT_SOURCE_URL, header=2, na_filter=False, skipinitialspace=True)

population_query = dw.query(
	'fryanpan13/covid-tracking-racial-data', 
    'SELECT * FROM population_data')
population_df = population_query.dataframe

population_df = population_df.rename(columns={'state': 'State', 'state_name': 'State Name', 'race': 'Race / Ethnicity', 'dataset': 'Dataset', 'geo_state_name': 'Geo State Name', 'population': 'Population' })

population_df['Population'] = population_df['Population'].astype('Int64')

population_index = ['Dataset', 'State', 'Race / Ethnicity']
population_df = population_df.set_index(population_index)

# Reformat date column
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

# Get rid of extra spaces in column names
df = df.rename(columns=lambda x: x.strip().replace(' .1','.1').replace(' .2','.2'))

# Standardize some column names
df = df.rename(columns={'Unknown': 'Unknown Race', 
    'Unknown.1': 'Unknown Race.1', 
    'Unknown.2': 'Unknown Ethnicity.1', 
    'Unknown.3': 'Unknown Race.2',
    'Unknown.4': 'Unknown Ethnicity.2'})

# Unpivot
races = ['Total', 'Known White', 'Known Black', 'Known LatinX / Hispanic', 'Known Asian', 
    'Known AIAN', 'Known NHPI', 'Known Multiracial', 'Other', 'Unknown Race',
    'Known Hispanic', 'Known Non-Hispanic', 'Unknown Ethnicity'] 


data = []
for race in races:
    cases_col = race
    deaths_col = race + '.1'
    negatives_col = race + '.2'
    dataset = 'Ethnicity' if race in ['Known Hispanic', 'Known Non-Hispanic', 'Unknown Ethnicity'] else 'Race'
    race_df = df[['Date', 'State']]
    race_df[CASES] = series_to_int(df[cases_col])
    race_df[DEATHS] = series_to_int(df[deaths_col])
    race_df[NEGATIVES] = series_to_int(df[negatives_col])
    race_df['Race / Ethnicity'] = race
    race_df['Dataset'] = dataset
    data.append(race_df)

df = pd.concat(data, ignore_index=True)



# Sort so all data for each state & ethnicity is adjacent and increasing in date
df = df.sort_values(['Dataset', 'State', 'Race / Ethnicity', 'Date'])

def sameIndex(df, period):
    return (df['Dataset'] == df['Dataset'].shift(period)) & (df['State'] == df['State'].shift(period)) & (df['Race / Ethnicity'] == df['Race / Ethnicity'].shift(period)) 

# compute differences, within the same dataset, state, and race / ethnicity
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

white_df = df[(df['Race / Ethnicity'] == 'Known White') & (df['Dataset'] == 'Race')]
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
for source_metric in all_metrics:
    if source_metric is 'Population': # No point calculating population per population :P
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

        df[dest_metric] = df[source_metric] / df[population] * 100000
        df[source_lo] = df[source_metric].apply(confidence_interval_lo)
        df[source_hi] = df[source_metric].apply(confidence_interval_hi)
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
        df[disparity_significant] = metric_lo > baseline_hi or metric_hi < baseline_lo 

print("")

df.to_csv('crdt_output.csv', index=True)
client = dw.api_client()
client.upload_files('fryanpan13/covid-tracking-racial-data',files='crdt_output.csv')

