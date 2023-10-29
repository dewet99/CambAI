#!/bin/bash

# Define the URL of the .tar.gz file
URL="https://us.openslr.org/resources/12/test-clean.tar.gz"

# Define the target directory for extraction
TARGET_DIRECTORY="./datasets"

# Ensure the target directory exists
mkdir -p "$TARGET_DIRECTORY"

# Download the .tar.gz file
wget -O "$TARGET_DIRECTORY/test-clean.tar.gz" "$URL"

# Check if the download was successful
if [ $? -eq 0 ]; then
    echo "Download successful."

    # Extract the downloaded file
    tar -xzvf "$TARGET_DIRECTORY/test-clean.tar.gz" -C "$TARGET_DIRECTORY"

    # Check if the extraction was successful
    if [ $? -eq 0 ]; then
        echo "Extraction successful. Data is located in $TARGET_DIRECTORY."
    else
        echo "Extraction failed."
    fi
else
    echo "Download failed."
fi
