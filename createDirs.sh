#!/bin/bash
clear
echo 'creating a folder for each ensemble'
for ensemble_folder in  {1..17}; do
	mkdir ensemble_number"$ensemble_folder";

done

