#!/bin/bash

input_file="breast-cancer-wisconsin.data"
output_file="breast-cancer-wisconsin-converted.data"

awk 'BEGIN{FS=OFS=","} {if ($NF == 2) $NF = 0; else if ($NF == 4) $NF = 1; print}' $input_file > $output_file

echo "Conversion complete. Output written to $output_file"
