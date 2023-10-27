#!/bin/bash

# Directory to monitor
monitor_dir="./output_files/jsons"

# Python script for processing JSON file
process_script="save_audio.py"


# Function to process the JSON file
process_json() {
    json_file="$1"
    python3 "$process_script" "$1" < "$json_file" 
}

while true; do
    # List JSON files in the directory
    json_files=("$monitor_dir"/*.json)

    if [ ${#json_files[@]} -gt 0 ]; then
        # At least one JSON file is present
        for json_file in "${json_files[@]}"; do
            # Process the JSON file
            process_json "$json_file"
            echo "Processed $json_file"
            
            # Optionally, you can remove the processed file
            rm "$json_file"
        done
    fi

    # Sleep for a short interval before checking again
    sleep 5
done
