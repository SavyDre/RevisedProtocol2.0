"""Configuration-loading and small validation helpers for the analysis entry point."""
import ast
import pandas as pd

def extract_max(python_list):
    """
    Extract the maximum value from a stringified Python list.
    Parameters:
    - python_list (str): A string representation of a Python list containing numeric values.
    Returns:
    - int: The maximum value in the list, converted to an integer.
    - None: If the input is invalid, not a list, or the list is empty.
    """
    try:
        # Safely evaluate the string as a Python literal
        python_list = ast.literal_eval(python_list)

        # Check if the evaluated result is a list and it's not empty
        if isinstance(python_list, list) and python_list:
            # Extract and return the maximum value as an integer
            return int(max(python_list))
        else:
            # Return None if the result is not a valid list or the list is empty
            return None
    except (ValueError, SyntaxError, TypeError):
        # Handle invalid strings or evaluation errors
        return None


def run_script_and_call(script_code, function_name='inputVariables'):
    """
    :param script_code: The object code contained within a script
    :param function_name: The name of the function defined within the script
    :return:
    """
    # Execute trusted configuration code in its own namespace (this is not a sandbox).
    namespace = {}

    # Execute the script (defines the function)
    exec(script_code, namespace)

    # Get the function and call it
    func = namespace.get(function_name)

    #
    if callable(func):
        return func()
    else:
        raise ValueError(f"'{function_name}' is not defined or not a function.")


def run_script_file_and_call(script_path, function_name='inputVariables'):
    """
    :param script_path: The path of the script.
    :param function_name: The name of the function defined within the script.
    :return:
    """
    #
    with open(script_path, 'r', encoding='utf-8-sig') as f:
        code = f.read()
    return run_script_and_call(code, function_name)

def reset_index(df, freq="min"):
    # Step: Get the current datetime index
    old_index = df.index

    # Step: Generate a new datetime index of same length and frequency
    new_index = pd.date_range(start=old_index[0], periods=len(df), freq=freq)

    # Step: Assign it as the new index
    df.index = new_index
    return df


def validate_frequency(suppliedfreq, referencefreq):
    """
        Validate a positive fixed sampling step no longer than the reference step.

        Parameters
        ----------
        suppliedfreq : str
            Supplied frequency string (e.g., "10s", "15min", "h").
        referencefreq : str
            Reference frequency string (e.g., "s", "min", "h", "D").

        Returns
        -------
        bool
            True if 0 < supplied duration <= reference duration.
            False otherwise, with a printed warning.
    """
    try:
        # Convert both frequencies to seconds
        supfreqdur = pd.tseries.frequencies.to_offset(suppliedfreq).nanos / 1e9
        reffreqdur = pd.tseries.frequencies.to_offset(referencefreq).nanos / 1e9

        # Compare
        if 0 < supfreqdur <= reffreqdur:
            return True
        else:
            print(f"Unsupported frequency: {suppliedfreq}. Provided frequency must be less or equal to {referencefreq}.")
            return False

    except Exception as e:
        print(f"Error validating frequency: {e}")
        return False
