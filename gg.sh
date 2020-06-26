#!/usr/bin/bash

echo "doing kdak1"
python3 cross_correlation.py II KDAK BHZ 2019-001 2020-170 -d 600 -o xxxx -k  > kdak1.txt
echo "doing kdak2"
python3 cross_correlation.py II KDAK BHZ 2015-001 2019-001 -d 600 -o xxxx -k  > kdak2.txt
echo "doing kdak3"
python3 cross_correlation.py II KDAK BHZ 2010-001 2015-001 -d 600 -o xxxx -k  > kdak3.txt
echo "doing kdak4"
python3 cross_correlation.py II KDAK BHZ 2005-001 2010-001 -d 600 -o xxxx -k  > kdak4.txt
echo "doing kdak5"
python3 cross_correlation.py II KDAK BHZ 2000-001 2005-001 -d 600 -o xxxx -k  > kdak5.txt
echo "doing kdak6"
python3 cross_correlation.py II KDAK BHZ 1997-169 2000-001 -d 600 -o xxxx -k  > kdak6.txt

cat kdak*.txt | awk '{print $2,$6,$8}' | grep -v ^II | grep -v ^IU | sort > DATA.II/KDAK.txt

exit



python3 cross_correlation.py II KDAK BHZ 2011-001 2012-001 -d 600 -o xxxx -k  > sur1.txt
