# Hole Detector and Counter

This application detects and counts holes in images using OpenCV and displays the results in a Streamlit web interface.

## Features

- Upload images or use a camera to capture images
- Detect holes in the image using computer vision techniques
- Display bounding boxes around detected holes with numbered labels
- Interactive selection - click on a row in the table to highlight the corresponding hole in the image
- View detailed information about each hole in a table format including:
  - Basic measurements (position, width, height, area)
  - Shape analysis (circularity, aspect ratio, perimeter)
  - Advanced features (equivalent diameter, solidity, orientation)
  - Spatial measurements (distance to image edge, distance between holes)
  - Helpful tooltips explaining each measurement
- Real-world measurements through size calibration
- Export hole data to CSV file for further analysis
- Save the annotated image
- Supports JPEG, PNG, and WebP image formats
- QR code for easy mobile access to the application

## Quick Start (Windows)

1. Run the `setup_environment.bat` script to create a virtual environment and install dependencies
2. Run the `run_app.bat` script to start the application
3. (Optional) Run `setup_git.bat` to initialize Git version control for the project

## Manual Installation

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   streamlit run hole_detector.py
   ```

3. Access from mobile devices by scanning the QR code displayed in the sidebar

## Version Control

The project includes a script to set up Git version control:

1. Run `setup_git.bat` to initialize the Git repository
2. The script creates an appropriate `.gitignore` file and makes the initial commit
3. To connect to a remote repository, use:
   ```
   git remote add origin [your-repository-url]
   git push -u origin master
   ```

## Usage

1. Select your input method (upload image or use camera)
2. If uploading, select an image file (JPG, PNG, or WebP)
3. If using camera, click "Take Photo"
4. View the detected holes with bounding boxes
5. Review hole details in the table below the image
   - Click on a row in the table to highlight the corresponding hole in the image
   - Detailed information about the selected hole will be displayed below the table
   - Use the "Column Descriptions" section to understand each measurement
6. For real-world measurements:
   - Enable size calibration
   - Select a reference hole
   - Enter the known diameter of that hole in millimeters
7. Save the resulting image or hole data as needed

## Customization

You can adjust the hole detection parameters in the `detect_holes` function:
- Adjust the threshold value (currently 100) to better detect holes in your specific images
- Modify the minimum area (currently 100 pixels) to filter out noise or include smaller holes

## Understanding Hole Metrics

- **Circularity**: Ranges from 0 to 1, with 1 being a perfect circle
- **Aspect Ratio**: Width divided by height, values > 1 indicate wider than tall holes
- **Solidity**: Area divided by convex hull area, measures convexity (1 = fully convex)
- **Orientation**: Angle of the major axis in degrees (0-180°)
- **Equivalent Diameter**: Diameter of a circle with same area as the hole
- **Distance to Nearest Hole**: Distance from this hole to the closest neighboring hole

## Real-World Measurements

With calibration enabled, the application calculates:
- Dimensions in millimeters (width, height, diameter)
- Area in square millimeters
- Perimeter in millimeters
- Distances in millimeters (to edge and between holes)

This provides accurate measurements for industrial applications where precise hole dimensions are required.
