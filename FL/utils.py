import csv

def write_csv(list):
    f = open('output.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    
    for l in list:
        wr.writerow(l)
    f.close()