import json

# Load the original JSON data
with open('../data/raw_course.json', 'r') as file:
    data = json.load(file)

# Filter the data for courses where the Site is "Coursera"
coursera_courses = [course for course in data if course.get('Site') == 'Coursera']

# Save the filtered data to a new JSON file
with open('../data/course.json', 'w') as outfile:
    json.dump(coursera_courses, outfile, indent=4)
