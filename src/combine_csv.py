import csv

csv_files = ['data/pbp-2013.csv', 'data/pbp-2014.csv', 'data/pbp-2015.csv', 'data/pbp-2016.csv', 'data/pbp-2017.csv']

master_csv = open("nfl_plays.csv", "a")

for element in csv_files:
    csv_file = open(element)
    csv_file.next()
    for line in csv_file:
        master_csv.write(line)
