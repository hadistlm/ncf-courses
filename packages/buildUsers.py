# Load Library
import orjson as json

# Load users from JSON
async def load_users(file_path='./user.json'):
  try:
    with open(file_path, 'rb') as f:
      json_users = json.loads(f.read())
    
    # Ensure the data is a list of dictionaries
    if not isinstance(json_users, list):
      raise ValueError("Expected a list of users")

    return json_users
  except FileNotFoundError as e:
    print(f"User file not found: {e}")
    return []
  except ValueError as e:
    print(f"Error loading user data: {e}")
    return []
  except Exception as e:
    print(f"Unexpected error loading users: {e}")
    return []