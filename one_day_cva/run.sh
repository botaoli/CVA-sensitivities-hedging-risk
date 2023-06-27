#!/bin/bash

# Specify the folder path
folder="params"
device="${2:-0}"

# Loop over each file in the folder
for file in "$folder"/*; do
    # Check if the item is a file (not a directory)
    if [ -f "$file" ]; then
        # Perform actions on the file
        echo "Using parameters: $file"
	runname=$(basename "$file")
	echo "python "new_CVA0.01_$1.py" $file $device > output/logs/$runname"_"$1"
	python "new_CVA0.01_$1.py" $file $device > output/logs/$runname"_"$1
        # Add your desired actions here
    fi
done

