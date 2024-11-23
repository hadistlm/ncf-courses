# Load Library
import orjson as json
# Load from helpers
from helpers.sanitizeNum import extract_numeric_value

# Load courses from JSON with error handling
async def load_courses(file_path='./course.json'):
  with open(file_path, 'rb') as f:
    json_courses = json.loads(f.read())

  # Ensure the data is a list of dictionaries
  if not isinstance(json_courses, list):
    raise ValueError("Expected a list of courses")
  
  courses = []
  for i, d in enumerate(json_courses):
    if not isinstance(d, dict):
      print(f"Unexpected course data format at index {i}: {d}")  # Debugging line to catch bad format
      continue  # Skip invalid data
    try:
      rating_str = str(d.get('Rating', '')).replace('stars', '').strip()
      rating = float(rating_str) if rating_str else 0.0
      viewers_str = str(d.get('Number of viewers', '')).strip()
      view = extract_numeric_value(viewers_str)

      courses.append({
        'id': d.get('ID', f'unknown_id_{i}'),  # Provide a fallback ID if missing
        'category': d.get('Category', 'Unknown'),
        'link': d.get('URL', '-'),
        'rating': rating,
        'view': view,
        'title': d.get('Title', 'Unknown'),
      })
    except ValueError as e:
      print(f"Error processing course {d.get('Title', 'Unknown')} at index {i}: {e}")
    except Exception as e:
      print(f"Unexpected error processing course {d} at index {i}: {e}")  # Catch all errors

  return courses