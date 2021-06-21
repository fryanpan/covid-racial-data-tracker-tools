from typing import Optional
import threading
import csv
from datetime import datetime as dt
import time
import re
from bs4 import BeautifulSoup
import scrapy


def cleanup(text: Optional[str]) -> Optional[str]:
    if text is None:
        return text

    return text.replace('\u200b','')\
               .replace('\xa0', ' ')\
               .replace('\r', ' ')\
               .replace('\n', ' ')\
               .replace('  ', ' ')\
               .strip()

class CaliforniaDataScraper(scrapy.Spider):
    name = 'ca-race-ethnicity'

    HEADER_AGE_RANGE_MAP = {
        'Cases and Deaths associated with COVID-19 by Race and Ethnicity': 'All',
        'All Cases and Deaths associated with COVID-19 by Race and Ethnicity': 'All',
        'All Cases and Deaths by Race and Ethnicity Among Ages 18+': '18+',
        'Proportions of Cases and Deaths by Race and Ethnicity Among Ages 0‐17': '0_17',
        'Proportions of Cases and Deaths by Race and Ethnicity Among Ages 18‐34': '18_34',
        'Proportions of Cases and Deaths by Race and Ethnicity Among Ages 18‐49': '18_49',
        'Proportions of Cases and Deaths by Race and Ethnicity Among Ages 35‐49': '35_49',
        'Proportions of Cases and Deaths by Race and Ethnicity Among Ages 50‐64': '50_64',
        'Proportions of Cases and Deaths by Race and Ethnicity Among Ages 65+': '65+',
        'Proportions of Cases and Deaths by Race and Ethnicity Among Ages 65-79': '65_79',
        'Proportions of Cases and Deaths by Race and Ethnicity Among Ages 80+': '80+'
    }

    # Matches shared.py RaceEthnicity Enum
    RACE_ETHNICITY_MAP = {
        'African American': 'Black',
        'African American/Black': 'Black',
        'American Indian': 'AIAN',
        'American Indian or Alaska Native': 'AIAN',
        'Asian': 'Asian',
        'Black': 'Black',
        'Latino': 'LatinX',
        'Multi-Race': 'Multiracial',
        'Native Hawaiian and other Pacific Islander': 'NHPI',
        'Native Hawaiian or Pacific Islander': 'NHPI',
        'Native Hawaiianand other Pacific Islander': 'NHPI',
        'Other': 'Other',
        'Total': 'Total',
        'Total with data': 'Total',
        'White': 'White'
    }

    def start_requests(self):
        yield scrapy.Request('https://www.cdph.ca.gov/Programs/CID/DCDC/Pages/COVID-19/Race-Ethnicity.aspx')

    def parse(self, response: scrapy.http.Response):
        timestamp = 0
        archive_date = '2000-01-01'
        date = '2000-01-01'
        date_pattern = "(January|February|March|April|May|June|July|August|September|October|November|December) [1-9][0-9]?, 202[0-9]"

        if 'wayback_machine_time' in response.meta:
            page_time = response.meta['wayback_machine_time']
            timestamp = page_time.timestamp()
            archive_date = page_time.isoformat()

        soup = BeautifulSoup(response.text, 'lxml')
        data = []

        date_title = soup.find('span', 'article-date-title')
        if date_title:
            date = date_title.text.strip(' \r\n')
            date = dt.strptime(date, '%m/%d/%Y').strftime('%Y-%m-%d')
            print(f'Scraping {date}')
        else:
            top_content = cleanup(soup.find('div', 'NewsItemContent').text)

            result = re.search(date_pattern, top_content)
            if result is not None:
                date = result.group(0)
                date = dt.strptime(date, '%B %d, %Y').strftime('%Y-%m-%d')            
                print(f'Scraping {date}')

        for table in soup.find_all('table', 'ms-rteTable-4'):
            prev_sibling = table.find_previous('h3')

            if 'ases' not in cleanup(prev_sibling.text): # old format - it's a date!
                prev_sibling = prev_sibling.find_previous('h3')

            header = cleanup(prev_sibling.text)
            print(f'header={repr(header)}')

            if 'All Cases and Deaths associated with COVID-19 by Race and Ethnicity' in header: # old format
                age_range = 'All'
            elif 'Cases and Deaths Associated with COVID-19 by Age Group in California' in header: # age group totals
                continue
            elif header not in CaliforniaDataScraper.HEADER_AGE_RANGE_MAP:
                raise Exception(f'Unknown header: {repr(header)}')
            else:
                age_range = CaliforniaDataScraper.HEADER_AGE_RANGE_MAP[header]

            for row in table.find_all('tr'):
                fields = row.find_all('td')

                if len(fields) == 0: # header row
                    continue

                race_ethnicity = cleanup(fields[0].text)
                if race_ethnicity:
                    race_ethnicity = CaliforniaDataScraper.RACE_ETHNICITY_MAP[race_ethnicity]

                values = [float(field.text.replace('\u200b','').replace(',','')) for field in fields[1:]]
                cases = int(values[0])
                deaths = int(values[2])

                row = [ timestamp, archive_date, date, age_range, race_ethnicity, cases, deaths ]
                data.append(row)
                
        return {
            'timestamp': timestamp,
            'date': date,
            'data': data
        }
