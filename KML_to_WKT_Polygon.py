import os
import numpy as np
from xml.etree import ElementTree as ET
from shapely.geometry import Polygon

def kml_to_wkt_polygon(directory, kml_dict, BoundingZ_max):
    """
    Converts coordinates from KML files in a directory to WKT polygon format
    and assigns them as values to corresponding keys in a dictionary.

    Parameters:
    - directory (str): Path to the directory containing the KML files.
    - kml_dict (dict): Dictionary where keys are KML file names (without extension),
      and values will be updated with details of WKT polygons.
    - BoundingZ_max (float): Maximum bounding elevation to use.

    Returns:
    - dict: Updated dictionary where each key contains:
        - "BoundingXY" (str): WKT polygon representation of the shape.
        - "BoundingZ" (list): Elevation bounds [mean elevation, BoundingZ_max].
        - "CentroidCoord" (list): Coordinates of the centroid [latitude, longitude].
    """
    updated_dict = {}

    for key in kml_dict:
        # Add the .kml extension to the key to form the filename
        filename = f"{key}.kml"
        file_path = os.path.join(directory, filename)

        if os.path.exists(file_path):
            # Parse the XML structure of the KML file
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()

                # Determine the namespace from the KML file
                namespace = {"kml": root.tag.split("}")[0].strip("{")}

                # Extract coordinates from the <coordinates> tag
                coordinates_tag = root.find(".//kml:coordinates", namespace)
                if coordinates_tag is not None:
                    # Parse the coordinates string
                    coordinates = coordinates_tag.text.strip().split()

                    # Convert coordinates to (latitude, longitude) pairs
                    lat_lon_pairs = [(float(coord.split(',')[1]), float(coord.split(',')[0])) for coord in coordinates]

                    # Calculate the mean elevation and define bounding elevations
                    elevations = [float(coord.split(',')[2]) for coord in coordinates]
                    BoundingZ = [np.mean(elevations), BoundingZ_max]

                    # Create a WKT polygon using Shapely
                    polygon = Polygon(lat_lon_pairs)
                    wkt_polygon = polygon.wkt  # Convert to WKT format

                    # Get the centroid of the polygon
                    centroid = polygon.centroid

                    # Update the dictionary with the WKT polygon and additional data
                    updated_dict[key] = {"BoundingXY": wkt_polygon,
                                         "BoundingZ": BoundingZ,
                                         "ReferenceCoord": [centroid.x, centroid.y]}
                else:
                    print(f"Coordinates not found in file: {filename}")
                    updated_dict[key] = None
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                updated_dict[key] = None
        else:
            print(f"File not found: {file_path}")
            updated_dict[key] = None

    return updated_dict
