import os
import orjson as json
import random

# Read the original JSON data from a file
file_path = os.path.join(os.path.dirname(__file__), '../data/raw_user.json')
with open(file_path, 'rb') as f:
    data = json.loads(f.read())

# Ensure the loaded data is a list
if isinstance(data, list):
    transformed_list = []
    
    # Loop through each entry in the list
    for entry in data[:1000]:
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
    with open(os.path.join(os.path.dirname(__file__), '../data/user.json'), 'wb') as outfile:
        outfile.write(json.dumps(transformed_list, option=json.OPT_INDENT_2))

    print("Data transformed and saved to user.json")
else:
    print("Error: Loaded data is not a list.")
