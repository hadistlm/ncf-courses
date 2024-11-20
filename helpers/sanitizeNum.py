import re

def extract_numeric_value(text):
  if isinstance(text, (str, int, float)):
    # If it's a number (int or float), convert it directly to int
    if isinstance(text, (int, float)):
        return int(float(text))  # Convert float to int correctly
      
    # Convert text to string and remove commas and periods
    text = str(text).replace(',', '').replace('.', '')
    
    # Find all numeric sequences in the modified string
    numbers = re.findall(r'\d+', text)
    
    # Return the first found number as an integer, or 0 if none found
    return int(numbers[0]) if numbers else 0
  else:
    print(f"Unexpected input to extract_numeric_value: {text}")
    return 0