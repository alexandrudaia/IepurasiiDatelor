#!/bin/bash

echo "Define   what ensemble to run(TYPE NAME): "
read ensemble_name
#echo "Define file from ENGINEERING FACTORY YOU WHANT To USE (TYPE NAME ) : "
# go back   where is list of  ensebmles
cd ..

#replace the  first occurence of  train.csv  and  test.csv  with path  from
#repository
sed -i s/train.csv/FEATURE_ENGINEERING_FACTORY/ $ensemble_name 
sed -i  s/test.csv/FEATURE_ENGINEERING_FACTORY/ $ensemble_name 
head -20  $ensemble_name
