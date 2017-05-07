#!/bin/bash
clear
echo " creating scoring  files "

for folder in {1..17}; do
	cd  ensemble_number"$folder";
	touch scores.txt;
	
done
