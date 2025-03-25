import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
import math
import socket
import qrcode
from PIL import Image
import io
from scipy.spatial.distance import cdist
import base64

# Hide the Streamlit footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def detect_holes(image):
    """
    Detect holes in an image using OpenCV
    Returns the original image with annotations, the count of holes, and hole details
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply binary thresholding
    _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area to avoid noise
    min_area = 100  # Adjust as needed
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Create a copy of the original image for drawing
    result_image = image.copy()
    
    # Prepare data for the table
    hole_data = []
    
    # Store hole centers for distance calculation
    centers = []
    
    # Draw bounding boxes and numbers
    for i, contour in enumerate(valid_contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Put the hole number
        cv2.putText(result_image, f"#{i+1}", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Calculate center of the hole
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Store center for distance calculation
        centers.append((center_x, center_y))
        
        # Calculate area
        area = cv2.contourArea(contour)
        
        # Calculate perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate circularity: 4*pi*area/perimeter^2 (1 for perfect circle)
        circularity = 0
        if perimeter > 0:
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            
        # Calculate aspect ratio
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Calculate equivalent diameter
        equiv_diameter = math.sqrt(4 * area / math.pi) if area > 0 else 0
        
        # Calculate convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        # Calculate solidity
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # Calculate minimum enclosing circle
        (center_x_circle, center_y_circle), radius = cv2.minEnclosingCircle(contour)
        
        # Calculate orientation
        if len(contour) >= 5:  # need at least 5 points for ellipse fitting
            (x_ellipse, y_ellipse), (MA, ma), angle = cv2.fitEllipse(contour)
            orientation = angle
        else:
            orientation = 0
            
        # Calculate moments for additional analysis
        M = cv2.moments(contour)
        
        # Calculate Hu Moments (shape descriptors)
        hu_moments = cv2.HuMoments(M).flatten()
        
        # Calculate distance from image edge
        img_height, img_width = image.shape[:2]
        dist_to_edge = min(center_x, center_y, img_width - center_x, img_height - center_y)
        
        # Add hole data to the list (without mean color)
        hole_data.append({
            "Hole #": i + 1,
            "Position X": int(center_x),
            "Position Y": int(center_y),
            "Width": w,
            "Height": h,
            "Area (px²)": int(area),
            "Perimeter (px)": round(perimeter, 2),
            "Circularity": round(circularity, 3),
            "Aspect Ratio": round(aspect_ratio, 2),
            "Equivalent Diameter": round(equiv_diameter, 2),
            "Solidity": round(solidity, 3),
            "Orientation (°)": round(orientation, 1),
            "Distance to Edge (px)": int(dist_to_edge)
        })
    
    # Calculate distances between holes if more than one hole
    if len(centers) > 1:
        # Calculate pairwise distances between all centers
        distances = cdist(centers, centers)
        
        # For each hole, find the distance to the nearest other hole
        for i in range(len(hole_data)):
            # Set diagonal (distance to self) to infinity
            distances[i, i] = float('inf')
            # Find minimum distance to another hole
            min_dist = np.min(distances[i])
            # Add to hole data
            hole_data[i]["Distance to Nearest Hole (px)"] = int(min_dist)
    else:
        # If only one hole, set distance to nearest hole as N/A
        for hole in hole_data:
            hole["Distance to Nearest Hole (px)"] = "N/A"
    
    return result_image, len(valid_contours), hole_data

def save_image(image):
    """Save the processed image with a timestamp in the filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"holes_detected_{timestamp}.jpg"
    save_path = os.path.join("output", filename)
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return save_path

def save_data_to_csv(hole_data):
    """Save the hole data to a CSV file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hole_data_{timestamp}.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(hole_data)
    save_path = os.path.join("output", filename)
    df.to_csv(save_path, index=False)
    
    return save_path

def calculate_real_measurements(hole_data, reference_hole_index, known_diameter_mm):
    """
    Recalculate measurements based on a reference hole with known diameter
    """
    if not hole_data or reference_hole_index >= len(hole_data):
        return hole_data
    
    # Get the reference hole's equivalent diameter in pixels
    reference_diam_px = hole_data[reference_hole_index]["Equivalent Diameter"]
    
    # Calculate the conversion factor (mm per pixel)
    if reference_diam_px > 0:
        mm_per_pixel = known_diameter_mm / reference_diam_px
        
        # Update all measurements with real-world units
        for hole in hole_data:
            # Convert pixel measurements to mm
            hole["Width (mm)"] = round(hole["Width"] * mm_per_pixel, 2)
            hole["Height (mm)"] = round(hole["Height"] * mm_per_pixel, 2)
            hole["Area (mm²)"] = round(hole["Area (px²)"] * mm_per_pixel * mm_per_pixel, 2)
            hole["Perimeter (mm)"] = round(hole["Perimeter (px)"] * mm_per_pixel, 2)
            hole["Equivalent Diameter (mm)"] = round(hole["Equivalent Diameter"] * mm_per_pixel, 2)
            hole["Distance to Edge (mm)"] = round(hole["Distance to Edge (px)"] * mm_per_pixel, 2)
            
            # Convert distance to nearest hole if present and not N/A
            if "Distance to Nearest Hole (px)" in hole and hole["Distance to Nearest Hole (px)"] != "N/A":
                hole["Distance to Nearest Hole (mm)"] = round(hole["Distance to Nearest Hole (px)"] * mm_per_pixel, 2)
            else:
                hole["Distance to Nearest Hole (mm)"] = "N/A"
            
    return hole_data

def generate_qr_code_for_url(url):
    """Generate a QR code for the given URL"""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert PIL image to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return img_byte_arr

def get_network_url():
    """Get the network URL for the Streamlit app"""
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    port = "8501"  # Default Streamlit port
    
    return f"http://{ip_address}:{port}"

def highlight_selected_hole(image, contours, selected_hole_index):
    """
    Create a copy of the image with the selected hole highlighted
    """
    # Create a copy of the image for highlighting
    highlighted_image = image.copy()
    
    # Highlight the selected contour
    if 0 <= selected_hole_index < len(contours):
        # Draw the selected contour with a different color (red)
        cv2.drawContours(highlighted_image, [contours[selected_hole_index]], 0, (255, 0, 0), 3)
        
        # Get bounding rect and put a more visible label
        x, y, w, h = cv2.boundingRect(contours[selected_hole_index])
        
        # Draw a more prominent label
        cv2.putText(highlighted_image, f"Selected: #{selected_hole_index+1}", 
                    (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    
    return highlighted_image

# Simpler client-side camera capture
def client_side_camera():
    """
    Implements client-side camera capture using JavaScript
    Returns the captured image if successful
    """
    st.info("To use your device's camera: Allow camera access in your browser when prompted")
    
    # Create a simple camera interface with a capture button
    capture_col1, capture_col2 = st.columns(2)
    
    with capture_col1:
        # Store the last captured image in session state
        if 'captured_image' not in st.session_state:
            st.session_state.captured_image = None
            
        # Simple camera interface with manual image upload as fallback
        st.subheader("Take a photo or upload from device")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "webp"])
        
        if uploaded_file is not None:
            # Convert the uploaded file to an OpenCV image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            # Convert BGR to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.session_state.captured_image = image_rgb
            return image_rgb
        
        st.caption("If camera access fails, please use the file uploader instead")
    
    with capture_col2:
        st.subheader("Image preview")
        
        if st.session_state.captured_image is not None:
            st.image(st.session_state.captured_image, caption="Captured Image", width=300)
            
            if st.button("Use this image"):
                return st.session_state.captured_image
            
            if st.button("Clear"):
                st.session_state.captured_image = None
                st.experimental_rerun()
                
    # Provide instructions for mobile users
    st.info("""
    **Mobile users:** If you're accessing from a mobile device and your camera doesn't work automatically:
    1. Take a photo using your device's camera app
    2. Upload the photo using the file uploader above
    """)
            
    return None

def is_running_remotely():
    """
    Determine if the app is being accessed remotely.
    This function creates a more reliable way to detect remote access.
    """
    # Properly initialize the session state value if it doesn't exist
    if 'is_remote_user' not in st.session_state:
        st.session_state['is_remote_user'] = False
    
    # Return the current remote status
    return st.session_state.is_remote_user

def main():
    st.title("Hole Detector and Counter")
    
    # Create a sidebar for the input options
    st.sidebar.title("Input Options")
    
    # Generate QR code for network access
    network_url = get_network_url()
    qr_code = generate_qr_code_for_url(network_url)
    
    # Display QR code in the sidebar
    st.sidebar.subheader("Scan to access on mobile")
    st.sidebar.image(qr_code, caption=f"Network URL: {network_url}", width=200)
    
    # Add a remote access toggle - properly initialize the session state first
    st.sidebar.markdown("---")
    
    # Initialize the session state if needed (redundant but safe)
    if 'is_remote_user' not in st.session_state:
        st.session_state['is_remote_user'] = False
    
    # Now use the checkbox with the properly initialized value
    remote_access = st.sidebar.checkbox(
        "I'm accessing remotely (use my device camera)", 
        value=st.session_state['is_remote_user']
    )
    
    # Update the session state value
    st.session_state['is_remote_user'] = remote_access
    
    st.sidebar.markdown("---")
    
    input_option = st.sidebar.radio("Select Input Method:", ["Upload Image", "Use Camera"])
    
    image = None
    
    if input_option == "Upload Image":
        uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            # Convert the uploaded file to an OpenCV image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            # Convert BGR to RGB for display
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    else:  # Use Camera
        st.sidebar.text("Camera Settings")
        
        # Check if accessing from a remote device using our improved detection
        if is_running_remotely():
            # Display instructions
            st.sidebar.info("The app will use your device's camera")
            
            # Use client-side camera capture for remote access
            image = client_side_camera()
        else:
            # Use server-side camera for local access
            st.sidebar.info("Using the server's camera. If you're on a remote device, check 'I'm accessing remotely' above.")
            if st.sidebar.button("Take Photo"):
                # Initialize camera
                cam = cv2.VideoCapture(0)
                ret, frame = cam.read()
                if ret:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cam.release()
                else:
                    st.error("Failed to access the camera. Please check if it's connected properly.")
    
    # Process the image if it's available
    if image is not None:
        # Display original image
        st.subheader("Original Image")
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Process the image to detect holes
        result_image, hole_count, hole_data = detect_holes(image)
        
        # Store the contours for later highlighting
        if 'contours' not in st.session_state:
            # Re-detect contours to have them available for highlighting
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 100
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            st.session_state.contours = valid_contours
            st.session_state.image = image
        
        # Display results
        st.subheader(f"Detected Holes: {hole_count}")
        
        # Create a placeholder for the image that can be updated
        image_placeholder = st.empty()
        image_placeholder.image(result_image, caption="Detected Holes", use_column_width=True)
        
        # Display hole data table
        if hole_count > 0:
            st.subheader("Hole Details")
            
            # Create columns for calibration controls
            cal_col1, cal_col2, cal_col3 = st.columns([1, 1, 1])
            
            with cal_col1:
                enable_calibration = st.checkbox("Enable Size Calibration")
            
            if enable_calibration:
                with cal_col2:
                    reference_hole = st.number_input("Reference Hole #", 
                                                  min_value=1, 
                                                  max_value=hole_count, 
                                                  value=1)
                with cal_col3:
                    known_diameter = st.number_input("Known Diameter (mm)", 
                                                  min_value=0.1, 
                                                  value=10.0, 
                                                  step=0.1)
                
                # Apply calibration 
                hole_data = calculate_real_measurements(hole_data, reference_hole-1, known_diameter)
                
                # Add a note about calibration
                st.info(f"All measurements are calibrated based on Hole #{reference_hole} having a diameter of {known_diameter} mm.")
            
            # Define column descriptions for tooltips
            column_descriptions = {
                "Hole #": "Unique identifier for each detected hole",
                "Position X": "X-coordinate of the hole's center in pixels",
                "Position Y": "Y-coordinate of the hole's center in pixels",
                "Width": "Width of the bounding box in pixels",
                "Height": "Height of the bounding box in pixels",
                "Area (px²)": "Area of the hole in square pixels",
                "Area (mm²)": "Area of the hole in square millimeters (after calibration)",
                "Perimeter (px)": "Length of the hole's boundary in pixels",
                "Perimeter (mm)": "Length of the hole's boundary in millimeters (after calibration)",
                "Circularity": "How close the hole is to a perfect circle (1.0 = perfect circle)",
                "Aspect Ratio": "Width divided by height (> 1 means wider than tall)",
                "Equivalent Diameter": "Diameter of a circle with the same area as the hole in pixels",
                "Equivalent Diameter (mm)": "Diameter of a circle with the same area in millimeters",
                "Solidity": "Ratio of contour area to its convex hull area (1.0 = fully convex)",
                "Orientation (°)": "Angle of the major axis in degrees (0-180°)",
                "Distance to Edge (px)": "Distance from hole center to the nearest image edge in pixels",
                "Distance to Edge (mm)": "Distance from hole center to the nearest image edge in millimeters",
                "Distance to Nearest Hole (px)": "Distance to the closest neighboring hole in pixels",
                "Distance to Nearest Hole (mm)": "Distance to the closest neighboring hole in millimeters"
            }
            
            # Display data table
            df = pd.DataFrame(hole_data)
            
            # Show the dataframe (removed help parameter)
            st.dataframe(df, use_container_width=True)
            
            # Instead, display column descriptions in an expandable section
            with st.expander("Column Descriptions"):
                for col, desc in column_descriptions.items():
                    if col in df.columns:
                        st.markdown(f"**{col}**: {desc}")
            
            # Add CSV download option
            if st.button("Save Hole Data as CSV"):
                csv_path = save_data_to_csv(hole_data)
                st.success(f"Hole data saved to {csv_path}")
        
        # Save image option
        if st.button("Save Result Image"):
            save_path = save_image(result_image)
            st.success(f"Image saved successfully at {save_path}")

if __name__ == "__main__":
    main()

