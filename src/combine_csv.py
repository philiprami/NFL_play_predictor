import csv

csv_files = ['../data/2013_weather.csv', '../data/2014_weather.csv', '../data/2015_weather.csv', '../data/2016_weather.csv', '../data/2017_weather.csv']

master_csv = open("../data/weather.csv", "a")

for element in csv_files:
    csv_file = open(element)
    csv_file.next()
    for line in csv_file:
        master_csv.write(line)
