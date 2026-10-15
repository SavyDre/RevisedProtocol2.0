"""Read and write dictionaries used for emission-source spatial mappings."""
import json

def save_dict_to_json(dictionary, file_path):
    """
    Save a dictionary as a JSON file.

    Parameters:
    - dictionary (dict): The dictionary to save.
    - file_path (str): The path of the file to save the JSON data.

    Returns:
    - None
    """
    try:
        with open(file_path, 'w') as json_file:
            json.dump(dictionary, json_file, indent=4)
        print(f"Dictionary successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving dictionary to JSON: {e}")


def load_dict_from_json(file_path):
    """Read a source-map object; invalid files fail before spatial classification."""
    with open(file_path, 'r', encoding='utf-8-sig') as json_file:
        dictionary = json.load(json_file)
    if not isinstance(dictionary, dict):
        raise ValueError('The emission-source mapper must be a JSON object.')
    return dictionary

