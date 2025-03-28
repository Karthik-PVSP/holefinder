import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import cv2
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import qrcode
import os
import math
from scipy import ndimage
from scipy.spatial.distance import cdist
from datetime import datetime
import io

class HoleDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hole Detector and Counter")
        self.root.geometry("1200x800")
        
        # Variables
        self.image = None
        self.processed_image = None
        self.uploaded_file = None
        self.hole_data = []
        self.valid_contours = []
        self.selected_hole_index = None
        
        # Create main frame
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create sidebar frame
        sidebar_frame = ttk.LabelFrame(main_frame, text="Controls")
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # File upload button
        upload_btn = ttk.Button(sidebar_frame, text="Upload Image", command=self.upload_file)
        upload_btn.pack(fill=tk.X, padx=10, pady=10)
        
        # Camera button (if available)
        camera_btn = ttk.Button(sidebar_frame, text="Use Camera", command=self.use_camera)
        camera_btn.pack(fill=tk.X, padx=10, pady=10)
        
        # Processing button
        process_btn = ttk.Button(sidebar_frame, text="Detect Holes", command=self.process_image)
        process_btn.pack(fill=tk.X, padx=10, pady=10)
        
        # Calibration frame
        calibration_frame = ttk.LabelFrame(sidebar_frame, text="Size Calibration")
        calibration_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.calibration_var = tk.BooleanVar(value=False)
        calibration_cb = ttk.Checkbutton(calibration_frame, text="Enable Calibration", 
                                         variable=self.calibration_var,
                                         command=self.toggle_calibration)
        calibration_cb.pack(anchor=tk.W, padx=10, pady=5)
        
        self.ref_hole_frame = ttk.Frame(calibration_frame)
        ttk.Label(self.ref_hole_frame, text="Reference Hole #:").pack(side=tk.LEFT)
        self.ref_hole_var = tk.IntVar(value=1)
        self.ref_hole_spinbox = ttk.Spinbox(self.ref_hole_frame, from_=1, to=100, 
                                            textvariable=self.ref_hole_var,
                                            command=self.update_calibration)
        self.ref_hole_spinbox.pack(side=tk.LEFT, padx=5)
        
        self.known_diameter_frame = ttk.Frame(calibration_frame)
        ttk.Label(self.known_diameter_frame, text="Known Diameter (mm):").pack(side=tk.LEFT)
        self.known_diameter_var = tk.DoubleVar(value=10.0)
        self.known_diameter_spinbox = ttk.Spinbox(self.known_diameter_frame, from_=0.1, to=1000, 
                                                 increment=0.1, textvariable=self.known_diameter_var,
                                                 command=self.update_calibration)
        self.known_diameter_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Bind variable changes to update calibration
        self.ref_hole_var.trace_add("write", lambda *args: self.update_calibration())
        self.known_diameter_var.trace_add("write", lambda *args: self.update_calibration())
        
        # Initially hide calibration controls
        self.ref_hole_frame.pack_forget()
        self.known_diameter_frame.pack_forget()
        
        # Export options
        export_frame = ttk.LabelFrame(sidebar_frame, text="Export Options")
        export_frame.pack(fill=tk.X, padx=10, pady=10)
        
        save_img_btn = ttk.Button(export_frame, text="Save Processed Image", command=self.save_image)
        save_img_btn.pack(fill=tk.X, padx=10, pady=5)
        
        save_data_btn = ttk.Button(export_frame, text="Save Hole Data as CSV", command=self.save_data_to_csv)
        save_data_btn.pack(fill=tk.X, padx=10, pady=5)
        
        # QR code button removed
        
        # Main content frame with notebook for tabs
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Images tab
        images_tab = ttk.Frame(self.notebook)
        self.notebook.add(images_tab, text="Images")
        
        # Split images tab into two parts
        self.original_frame = ttk.LabelFrame(images_tab, text="Original Image")
        self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.original_image_label = ttk.Label(self.original_frame)
        self.original_image_label.pack(fill=tk.BOTH, expand=True)
        
        self.processed_frame = ttk.LabelFrame(images_tab, text="Detected Holes")
        self.processed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.processed_image_label = ttk.Label(self.processed_frame)
        self.processed_image_label.pack(fill=tk.BOTH, expand=True)
        
        # Hole Data tab
        data_tab = ttk.Frame(self.notebook)
        self.notebook.add(data_tab, text="Hole Data")
        
        # Create treeview for hole data
        self.hole_treeview_frame = ttk.Frame(data_tab)
        self.hole_treeview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create treeview with scrollbars
        treeview_scroll_y = ttk.Scrollbar(self.hole_treeview_frame)
        treeview_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        treeview_scroll_x = ttk.Scrollbar(self.hole_treeview_frame, orient='horizontal')
        treeview_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.hole_treeview = ttk.Treeview(self.hole_treeview_frame, 
                                          yscrollcommand=treeview_scroll_y.set,
                                          xscrollcommand=treeview_scroll_x.set)
        
        treeview_scroll_y.config(command=self.hole_treeview.yview)
        treeview_scroll_x.config(command=self.hole_treeview.xview)
        
        self.hole_treeview.pack(fill=tk.BOTH, expand=True)
        
        # Configure treeview columns
        self.hole_treeview['columns'] = ('hole_num', 'pos_x', 'pos_y', 'width', 'height', 
                                         'area', 'perimeter', 'circularity')
        self.hole_treeview.column('#0', width=0, stretch=tk.NO)
        self.hole_treeview.column('hole_num', anchor=tk.CENTER, width=80)
        self.hole_treeview.column('pos_x', anchor=tk.CENTER, width=100)
        self.hole_treeview.column('pos_y', anchor=tk.CENTER, width=100)
        self.hole_treeview.column('width', anchor=tk.CENTER, width=80)
        self.hole_treeview.column('height', anchor=tk.CENTER, width=80)
        self.hole_treeview.column('area', anchor=tk.CENTER, width=100)
        self.hole_treeview.column('perimeter', anchor=tk.CENTER, width=100)
        self.hole_treeview.column('circularity', anchor=tk.CENTER, width=100)
        
        # Configure headings
        self.hole_treeview.heading('hole_num', text='Hole #')
        self.hole_treeview.heading('pos_x', text='Position X')
        self.hole_treeview.heading('pos_y', text='Position Y')
        self.hole_treeview.heading('width', text='Width')
        self.hole_treeview.heading('height', text='Height')
        self.hole_treeview.heading('area', text='Area (px²)')
        self.hole_treeview.heading('perimeter', text='Perimeter (px)')
        self.hole_treeview.heading('circularity', text='Circularity')
        
        # Bind treeview selection to highlight the selected hole
        self.hole_treeview.bind('<<TreeviewSelect>>', self.on_hole_selected)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def toggle_calibration(self):
        """Show or hide calibration controls based on checkbox state"""
        if self.calibration_var.get():
            self.ref_hole_frame.pack(fill=tk.X, padx=10, pady=5)
            self.known_diameter_frame.pack(fill=tk.X, padx=10, pady=5)
            # Update calibration if we have hole data
            if self.hole_data:
                self.update_calibration()
        else:
            self.ref_hole_frame.pack_forget()
            self.known_diameter_frame.pack_forget()
            # Reset to pixel-based measurements if we disable calibration
            if self.hole_data and self.processed_image is not None:
                self.process_image()
    
    def upload_file(self):
        """Open file dialog to select and load an image"""
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.webp")]
        )
        
        if file_path:
            self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
            self.uploaded_file = file_path
            self.image = cv2.imread(file_path)
            self.display_original_image()
            self.hole_data = []
            self.valid_contours = []
            self.clear_treeview()
    
    def use_camera(self):
        """Capture image from camera"""
        try:
            # Initialize camera
            cam = cv2.VideoCapture(0)
            ret, frame = cam.read()
            
            if ret:
                self.image = frame
                self.display_original_image()
                self.hole_data = []
                self.valid_contours = []
                self.clear_treeview()
                self.status_var.set("Image captured from camera")
            else:
                messagebox.showerror("Camera Error", "Failed to capture from camera")
            
            # Release camera
            cam.release()
        except Exception as e:
            messagebox.showerror("Camera Error", f"Error accessing camera: {str(e)}")
    
    def display_original_image(self):
        """Display the original image in the UI"""
        if self.image is not None:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            # Resize for display while maintaining aspect ratio
            image_rgb = self.resize_for_display(image_rgb)
            # Convert to PhotoImage
            img = Image.fromarray(image_rgb)
            photo = ImageTk.PhotoImage(image=img)
            
            # Update label
            self.original_image_label.configure(image=photo)
            self.original_image_label.image = photo  # Keep a reference
    
    def resize_for_display(self, image, max_size=400):
        """Resize image for display while maintaining aspect ratio"""
        h, w = image.shape[:2]
        if h > w:
            new_h = max_size
            new_w = int(w * (max_size / h))
        else:
            new_w = max_size
            new_h = int(h * (max_size / w))
        
        return cv2.resize(image, (new_w, new_h))
    
    def process_image(self):
        """Process the image to detect holes"""
        if self.image is None:
            messagebox.showerror("Error", "Please upload an image first!")
            return
        
        # Reset original hole data
        if hasattr(self, 'original_hole_data'):
            delattr(self, 'original_hole_data')
        
        # Detect holes
        result_image, hole_count, self.hole_data = self.detect_holes(self.image)
        self.processed_image = result_image
        
        # Apply calibration if enabled
        if self.calibration_var.get():
            # Store original hole data before calibration
            self.original_hole_data = self.hole_data.copy()
            
            ref_hole_index = self.ref_hole_var.get() - 1
            known_diameter = self.known_diameter_var.get()
            
            if 0 <= ref_hole_index < len(self.hole_data):
                self.hole_data = self.calculate_real_measurements(
                    self.hole_data, ref_hole_index, known_diameter)
            else:
                messagebox.showwarning("Calibration Warning", 
                                      f"Reference hole #{ref_hole_index+1} not found. Calibration not applied.")
        
        # Display the processed image
        self.display_processed_image()
        
        # Update hole data in treeview
        self.update_hole_treeview()
        
        # Switch to Images tab
        self.notebook.select(0)
        
        # Update status
        self.status_var.set(f"Detected {hole_count} holes")
    
    def display_processed_image(self):
        """Display the processed image with detected holes"""
        if self.processed_image is not None:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
            # Resize for display
            image_rgb = self.resize_for_display(image_rgb)
            # Convert to PhotoImage
            img = Image.fromarray(image_rgb)
            photo = ImageTk.PhotoImage(image=img)
            
            # Update label
            self.processed_image_label.configure(image=photo)
            self.processed_image_label.image = photo  # Keep a reference
    
    def clear_treeview(self):
        """Clear all items from the treeview"""
        for item in self.hole_treeview.get_children():
            self.hole_treeview.delete(item)
    
    def update_hole_treeview(self):
        """Update the treeview with hole data"""
        # Clear existing data
        self.clear_treeview()
        
        # Configure columns based on available data
        sample_hole = self.hole_data[0] if self.hole_data else None
        if not sample_hole:
            return
        
        # Determine which columns to show
        columns = list(sample_hole.keys())
        
        # Configure treeview columns
        self.hole_treeview['columns'] = columns
        self.hole_treeview.column('#0', width=0, stretch=tk.NO)
        
        # Set column widths and headings
        for col in columns:
            self.hole_treeview.column(col, anchor=tk.CENTER, width=100)
            self.hole_treeview.heading(col, text=col)
        
        # Add data to treeview
        for i, hole in enumerate(self.hole_data):
            values = [hole.get(col, '') for col in columns]
            self.hole_treeview.insert('', tk.END, text='', values=values, iid=i)
    
    def on_hole_selected(self, event):
        """Handle treeview selection to highlight the selected hole"""
        selection = self.hole_treeview.selection()
        if selection:
            item_id = selection[0]
            self.selected_hole_index = int(item_id)
            
            # Highlight the selected hole in the image
            if self.image is not None and 0 <= self.selected_hole_index < len(self.valid_contours):
                highlighted_image = self.highlight_selected_hole(
                    self.processed_image.copy(), 
                    self.valid_contours, 
                    self.selected_hole_index
                )
                
                # Display the highlighted image
                image_rgb = cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB)
                image_rgb = self.resize_for_display(image_rgb)
                img = Image.fromarray(image_rgb)
                photo = ImageTk.PhotoImage(image=img)
                
                self.processed_image_label.configure(image=photo)
                self.processed_image_label.image = photo
    
    def save_image(self):
        """Save the processed image to a file"""
        if self.processed_image is None:
            messagebox.showerror("Error", "No processed image to save!")
            return
        
        # Get current timestamp for default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"holes_detected_{timestamp}.png"
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_filename,
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            cv2.imwrite(file_path, self.processed_image)
            self.status_var.set(f"Image saved to {os.path.basename(file_path)}")
            messagebox.showinfo("Success", "Image saved successfully!")
    
    def save_data_to_csv(self):
        """Save the hole data to a CSV file"""
        if not self.hole_data:
            messagebox.showerror("Error", "No hole data to save!")
            return
        
        # Get current timestamp for default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"hole_data_{timestamp}.csv"
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default_filename,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            # Convert to DataFrame and save
            df = pd.DataFrame(self.hole_data)
            df.to_csv(file_path, index=False)
            
            self.status_var.set(f"Hole data saved to {os.path.basename(file_path)}")
            messagebox.showinfo("Success", "Hole data saved successfully!")
    
    # Core image processing functions
    def detect_holes(self, image):
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
        self.valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Create a copy of the original image for drawing
        result_image = image.copy()
        
        # Prepare data for the table
        hole_data = []
        
        # Store hole centers for distance calculation
        centers = []
        
        # Draw bounding boxes and numbers
        for i, contour in enumerate(self.valid_contours):
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
            
            # Calculate distance from image edge
            img_height, img_width = image.shape[:2]
            dist_to_edge = min(center_x, center_y, img_width - center_x, img_height - center_y)
            
            # Add hole data to the list
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
        
        return result_image, len(self.valid_contours), hole_data

    def calculate_real_measurements(self, hole_data, reference_hole_index, known_diameter_mm):
        """
        Recalculate measurements based on a reference hole with known diameter
        """
        if not hole_data or reference_hole_index >= len(hole_data):
            return hole_data
            
        # Make a copy of the hole data to avoid modifying the original
        calibrated_data = []
        for hole in hole_data:
            calibrated_data.append(hole.copy())
        
        # Get the reference hole's equivalent diameter in pixels
        reference_diam_px = calibrated_data[reference_hole_index]["Equivalent Diameter"]
        
        # Calculate the conversion factor (mm per pixel)
        if reference_diam_px > 0:
            mm_per_pixel = known_diameter_mm / reference_diam_px
            
            # Update all measurements with real-world units
            for hole in calibrated_data:
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
                
        return calibrated_data

    def highlight_selected_hole(self, image, contours, selected_hole_index):
        """
        Create a copy of the image with the selected hole highlighted
        """
        # Create a copy of the image for highlighting
        highlighted_image = image.copy()
        
        # Highlight the selected contour
        if 0 <= selected_hole_index < len(contours):
            # Draw the selected contour with a different color (red)
            cv2.drawContours(highlighted_image, [contours[selected_hole_index]], 0, (0, 0, 255), 3)
            
            # Get bounding rect and put a more visible label
            x, y, w, h = cv2.boundingRect(contours[selected_hole_index])
            
            # Draw a more prominent label
            cv2.putText(highlighted_image, f"Selected: #{selected_hole_index+1}", 
                        (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        return highlighted_image

    def update_calibration(self):
        """Update hole measurements based on current calibration settings"""
        if not self.hole_data or not self.calibration_var.get():
            return
            
        # Store original hole data if not already stored
        if not hasattr(self, 'original_hole_data'):
            self.original_hole_data = self.hole_data.copy()
        else:
            # Reset to original data before applying new calibration
            self.hole_data = self.original_hole_data.copy()
            
        # Get current calibration values
        try:
            ref_hole_index = self.ref_hole_var.get() - 1
            known_diameter = self.known_diameter_var.get()
            
            # Validate the reference hole index
            if 0 <= ref_hole_index < len(self.hole_data):
                # Apply the calibration
                self.hole_data = self.calculate_real_measurements(
                    self.hole_data, ref_hole_index, known_diameter)
                
                # Update the treeview
                self.update_hole_treeview()
                
                # Update status
                self.status_var.set(f"Calibration updated: Reference hole #{ref_hole_index+1}, diameter = {known_diameter} mm")
            else:
                # Only show a warning if we have hole data but the index is invalid
                if self.hole_data:
                    self.status_var.set(f"Warning: Reference hole #{ref_hole_index+1} not found. Calibration not applied.")
        except (ValueError, TclError):
            # Handle case where the spinbox values are being edited
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = HoleDetectorApp(root)
    root.mainloop()
