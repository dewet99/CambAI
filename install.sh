#!/bin/bash

# All necessary stuff needs to be downloaded into the following dir. On my PC this was /home/dewet/Documents/Camb
current_directory="$(pwd)"


###################################################################################################
###################################################################################################
# Funky code to select which python command to use, because I have like 3... :')
# Initialize the selected Python command variable
selected_python=""

# Define the list of Python commands to check
python_commands=("python3.10" "python3" "python")

# Loop through the Python commands
for cmd in "${python_commands[@]}"; do
    # Check if the Python command exists and is Python 3.10 or greater
    if command -v "$cmd" &> /dev/null && [[ "$("$cmd" --version 2>&1)" == *"Python 3."* ]]; then
        python_version="$("$cmd" --version 2>&1)"
        # Extract the Python version number
        python_version="${python_version#Python }"
        # Check if the version is greater than or equal to 3.10
        if $cmd -c "import sys; print(sys.version_info >= (3, 10))" | grep -q "True"; then
            selected_python="$cmd"
            break
        fi
    fi
done

# Check if a suitable Python version was found
if [[ -n "$selected_python" ]]; then
    echo "Selected Python: $selected_python"
    # Use the selected Python command throughout the rest of your script
    # For example, you can run your Python code with: "$selected_python your_script.py"
else
    echo "No suitable Python version (3.10 or greater) found."
fi 
###################################################################################################
###################################################################################################

# we get the serve git repo and the knn_vc git repo

git clone "https://github.com/bshall/knn-vc.git" "knn_vc"
git clone "https://github.com/pytorch/serve.git"


# we run the install_dependencies script from serve, because we need some of the dependencies to generate the state dicts. This is a lot more 
# effort than just providing the 
$select_python -m venv venv
source ./venv/bin/activate
python ./serve/ts_scripts/install_dependencies.py --cuda=cu118


# Build the torchserve docker image. Ideally we'd pass args to specify cuda version and python version, etc, but that's a nice-to-have for now
./serve/docker/build_image.sh -g -cv cu118 -py 3.10

# Download the state dicts for the knn_vc_model
mkdir knn-vc-downloads
mkdir state_dicts
$selected_python utils/get_state_dicts.py
