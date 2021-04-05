# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import csv

class CaliforniaPipeline:
    def open_spider(self, spider):
        print("open_spider")
        self.all_data = {}

    def close_spider(self, spider):
        print("close_spider, writing output")

        self.file = open(f'output.csv', 'w')
        self.data_writer = csv.writer(self.file)
        self.data_writer.writerow(['archive_timestamp', 'archive_date', 'date', 'age_range', 'race_ethnicity', 'cases', 'deaths'])

        dates = list(self.all_data.keys())
        dates.sort()
        for date in dates:
            for row in self.all_data[date]['data']:
                self.data_writer.writerow(row)

    def process_item(self, item, spider):
        print(f"process_item {item['date']}")
        date = item['date']
        if (date not in self.all_data) or (self.all_data[date]['timestamp'] < item['timestamp']):
            self.all_data[date] = item

        print(f"dates stored: {len(self.all_data)}")
        return item
