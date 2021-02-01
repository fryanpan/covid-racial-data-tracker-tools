from enum import Enum
import datadotworld as dw

class RaceEthnicity(Enum):
    TOTAL = 'Total'
    WHITE = 'White'
    BLACK = 'Black'
    AIAN = 'AIAN'
    NHPI = 'NHPI'
    LATINX = 'LatinX'
    ASIAN = 'Asian'
    MULTIRACIAL = 'Multiracial'
    OTHER = 'Other'
    HISPANIC = 'Hispanic'
    NON_HISPANIC = 'NonHispanic'
    UNKNOWN = 'Unknown'

def save_to_dw(df, filename):
    file_path = f'/tmp/{filename}'
    df.to_csv(file_path, index=True)
    #client = dw.api_client()
    #client.upload_files('fryanpan13/covid-tracking-racial-data',files=file_path)
