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
    """
    Read a JSON file into a dictionary.

    Parameters:
    - file_path (str): The path of the JSON file to read.

    Returns:
    - dict: The dictionary read from the JSON file.
    """
    try:
        with open(file_path, 'r') as json_file:
            dictionary = json.load(json_file)
        print(f"Dictionary successfully loaded from {file_path}")
        return dictionary
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return {}
