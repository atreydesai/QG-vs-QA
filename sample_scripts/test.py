import os
import re

def search_for_pkl(directory):
    """
    Recursively searches a directory and its subdirectories for files containing "pkl" in their content.

    Args:
        directory: The path to the directory to search.
    """

    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        return

    for root, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:  #encoding and error handling added.
                    content = f.read()
                    if re.search(r'\bpkl\b', content, re.IGNORECASE): # Use re.search for case-insensitive search and \b for word boundary.
                        print(f"Found 'pkl' in: {filepath}")
            except (UnicodeDecodeError, PermissionError) as e:
                print(f"Error processing '{filepath}': {e}")  # Handle encoding errors and permission issues
            except Exception as e:
                print(f"An unexpected error occurred while processing '{filepath}': {e}")


if __name__ == "__main__":
    target_directory = input("Enter the directory path: ")  # Get directory path from user
    search_for_pkl(target_directory)