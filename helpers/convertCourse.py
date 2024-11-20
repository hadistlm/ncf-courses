import os
import orjson as json

# Load the original JSON data
file_path = os.path.join(os.path.dirname(__file__), '../data/raw_course.json')
with open(file_path, 'rb') as f:
    data = json.loads(f.read())

# Filter the data for courses where the Site is "Coursera"
coursera_courses = [course for course in data if course.get('Site') == 'Coursera']

# Save the filtered data to a new JSON file
with open(os.path.join(os.path.dirname(__file__), '../data/course.json'), 'wb') as outfile:
    outfile.write(json.dumps(coursera_courses, option=json.OPT_INDENT_2))
