#!/bin/bash

# Download the zip file
wget https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip

# Unzip the file into the folder
unzip data_ids_april7.zip -d 2wiki

# Check if the unzip was successful
if [ $? -eq 0 ]; then
  echo "Unzip successful! Files are in the '2wiki' folder."
else
  echo "Unzip failed. Please check if 'data_ids_april7.zip' exists and is a valid zip file."
fi

# Remove the zip file
rm data_ids_april7.zip