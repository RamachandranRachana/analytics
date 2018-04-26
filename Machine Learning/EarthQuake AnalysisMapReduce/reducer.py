import sys
import math
from datetime import datetime
from decimal import *

maxi=0
outer_row=''
count=0
mag=0
inner_row=''
datelst=[]
maglst=[]
magl=0
for line in sys.stdin:
        outer_row=line.split(',')
        datelst.append(outer_row[0])
        maglst.append(outer_row[1])      
lendate=len(datelst)
date=datelst[0]
for i in range(0,lendate-1):
        if date==datelst[i]:
	    try:
               magl=float(maglst[i])
            except ValueError:
               continue
            mag+=magl
  	else:
            try:
               magl=float(maglst[i])
            except ValueError:
               continue
	    date=datelst[i]
	    mag+=magl
            count+=1
            if count==7:
               count=0
 	       avg=mag/7
	       print avg
print avg
