#!/bin/bash

#args:
#1 -> num runs to do
#2 -> model_name
#3 -> dataset_relative_path to script
#i -> id to be used in generating output files
#4 -> data_type: noise or audio

# Number of times to run the Python script
num_runs=$1  # Adjust this to the desired number of runs

container_url="http://localhost:8080/predictions/$2"

echo "Performing inference at $container_url"


# Loop to run the Python script multiple times with the same arguments
for ((i=1; i<=num_runs; i++)); do
    echo "Running Python script: Run $i with arguments: $1 $2"
    python3 generate_input.py $2 $3 $i $4 # convert the .flac files to json files in the correct format
    #curl "$container_url" -T "./input_data.json"  > "./output_files/jsons/response_$i.json" &
    curl "$container_url" -T "./input_data.json" > "/dev/null" &
    echo "Done generating and sending inference request $i"
done

exit 1
