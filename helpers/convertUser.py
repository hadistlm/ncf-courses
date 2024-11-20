import json
import random

# Read the original JSON data from a file
with open('../data/raw_user.json', 'r') as infile:
    data = json.load(infile)

# Ensure the loaded data is a list
if isinstance(data, list):
    transformed_list = []
    
    # Loop through each entry in the list
    for entry in data[:100]:
        if isinstance(entry, dict):  # Check if the entry is a dictionary
            # Randomly determine if interests should be empty (20% chance)
            if random.random() < 0.4:  # 40% chance
                interests = []
            else:
                interests = [entry["interest"], entry["sub_interest"]]

            transformed_data = {
                "id": entry["id"],
                "name": entry["first_name"] + " " + entry["last_name"],
                "interests": interests
            }
            transformed_list.append(transformed_data)

    # Write the transformed data to a new JSON file
    with open('../data/user.json', 'w') as outfile:
        json.dump(transformed_list, outfile, indent=2)

    print("Data transformed and saved to user.json")
else:
    print("Error: Loaded data is not a list.")
