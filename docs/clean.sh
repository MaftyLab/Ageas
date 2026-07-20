#!/bin/bash

echo "Cleaning up build artifacts..."

# Remove the HTML folder
if [ -d "html" ]; then
    if rm -rf "html"; then
        echo " -> Successfully removed html"
    else
        echo " -> Error: Failed to remove html"
    fi
fi

# Remove the doctrees folder
if [ -d "doctrees" ]; then
    if rm -rf "doctrees"; then
        echo " -> Successfully removed doctrees"
    else
        echo " -> Error: Failed to remove doctrees"
    fi
fi

# Remove the generated folder
if [ -d "generated" ]; then
    if rm -rf "generated"; then
        echo " -> Successfully removed generated"
    else
        echo " -> Error: Failed to remove generated"
    fi
fi

echo "Process finished!"
