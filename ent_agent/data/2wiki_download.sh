#!/bin/bash
if [ ! -d "datasets" ]; then
    mkdir datasets
fi

if [ ! -d "datasets/2wiki" ]; then
    rm -rf datasets/2wiki
fi


mkdir datasets/2wiki
mkdir datasets/2wiki_entity

wget https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip
# Define the folder name
FOLDER_NAME="./datasets/2wiki"

# Check if the folder exists, if not, create it
if [ ! -d "$FOLDER_NAME" ]; then
  mkdir -p "$FOLDER_NAME"
fi

# Unzip the file into the folder
unzip data_ids_april7.zip -d "$FOLDER_NAME"

# Check if the unzip was successful
if [ $? -eq 0 ]; then
  echo "Unzip successful! Files are in the '$FOLDER_NAME' folder."
else
  echo "Unzip failed. Please check if 'data_ids_april7.zip' exists and is a valid zip file."
fi

# Remove the zip file
rm data_ids_april7.zip