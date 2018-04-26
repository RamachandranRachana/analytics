import sys
#with open ('all_week.csv') as csvfile:
#    a = csv.reader(csvfile,delimiter=',')
#    for row in a:
#        date2=row[0].split('T')
#        mag2=row[4]
#        print date2[0]+','+mag2
for line in sys.stdin:
        #words=line.split('T')
        line=line.split(',')
        first = line[0].split('T')
        print first[0]+','+line[4]

