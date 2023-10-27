#!/bin/bash

# Number of times to run the Python script
num_runs=$3  # Adjust this to the desired number of runs

container_url="http://localhost:8080/predictions/$1"

echo "Performing inference at $container_url"


# Loop to run the Python script multiple times with the same arguments
for ((i=1; i<=num_runs; i++)); do
    echo "Running Python script: Run $i with arguments: $1 $2"
    python3 generate_input.py $1 $2 $i # convert the .flac files to json files in the correct format
    curl "$container_url" -T "./input_data.json"  > "./output_files/jsons/response_$i.json" &
done

exit 1
