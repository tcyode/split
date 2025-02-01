# split.py
from PyPDF2 import PdfReader, PdfWriter
import os
import re
from datetime import datetime
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import tempfile
import cv2
import numpy as np
import shutil

# Add debugging information
print("Script starting...")
print("Current working directory:", os.getcwd())

# Set Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def analyze_text_pattern(gray_image, x, y, w, h):
    """
    Analyze text patterns in a region to determine if it's part of a receipt.
    Returns: (is_receipt_like, has_numbers, alignment_type)
    """
    # Extract region
    roi = gray_image[y:y+h, x:x+w]
    
    # Threshold to get text
    _, binary = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Split into vertical sections for column analysis
    num_sections = 3
    section_width = w // num_sections
    sections = []
    
    for i in range(num_sections):
        start_x = i * section_width
        end_x = start_x + section_width
        section = binary[:, start_x:end_x]
        sections.append(section)
    
    # Analyze text density in each section
    densities = []
    for section in sections:
        text_pixels = np.sum(section == 255)
        density = text_pixels / (section.shape[0] * section.shape[1])
        densities.append(density)
    
    # Check for receipt-like patterns
    left_dense = densities[0] > 0.05  # Left section has text
    right_dense = densities[-1] > 0.05  # Right section has text
    middle_sparse = densities[1] < max(densities[0], densities[-1])  # Middle less dense
    
    # Detect if region contains numbers (likely prices)
    has_numbers = False
    if right_dense:
        # Use OCR on right section to check for number patterns
        right_roi = roi[:, -section_width:]
        text = pytesseract.image_to_string(right_roi, config='--psm 6')
        has_numbers = bool(re.search(r'\d+\.\d{2}', text))  # Price pattern
    
    # Determine alignment type
    if left_dense and right_dense and middle_sparse:
        alignment = 'dual-column'
    elif left_dense:
        alignment = 'left'
    elif right_dense:
        alignment = 'right'
    else:
        alignment = 'unknown'
    
    is_receipt_like = (left_dense or right_dense) and has_numbers
    
    return is_receipt_like, has_numbers, alignment

def should_merge_regions(region1, region2, gray_image, proximity_threshold_px):
    """
    Determine if two regions should be merged based on text patterns and alignment.
    """
    x1, y1, w1, h1 = region1
    x2, y2, w2, h2 = region2
    
    # Basic distance checks
    vertical_overlap = (min(y1 + h1, y2 + h2) - max(y1, y2)) > 0
    horizontal_overlap = (min(x1 + w1, x2 + w2) - max(x1, x2)) > 0
    
    # Calculate distances
    vertical_distance = (min(abs(y1 - (y2 + h2)), abs((y1 + h1) - y2)) 
                        if not vertical_overlap else 0)
    horizontal_distance = (min(abs(x1 - (x2 + w2)), abs((x1 + w1) - x2)) 
                         if not horizontal_overlap else 0)
    
    # Analyze text patterns in both regions
    pattern1 = analyze_text_pattern(gray_image, x1, y1, w1, h1)
    pattern2 = analyze_text_pattern(gray_image, x2, y2, w2, h2)
    
    # Check if regions are complementary (e.g., items and prices)
    complementary_columns = (
        (pattern1[2] == 'left' and pattern2[2] == 'right') or
        (pattern1[2] == 'right' and pattern2[2] == 'left')
    )
    
    # Check vertical alignment for potential columns
    vertical_alignment = abs(y1 - y2) < proximity_threshold_px
    
    # Determine if regions should be merged
    should_merge = (
        # Standard proximity checks
        (vertical_overlap and horizontal_distance < proximity_threshold_px * 5) or  # Allow wider gaps
        (horizontal_overlap and vertical_distance < proximity_threshold_px) or
        # Column-specific checks
        (complementary_columns and vertical_alignment and 
         horizontal_distance < proximity_threshold_px * 8)  # Even wider for columns
    )
    
    return should_merge

def merge_nearby_regions(regions, proximity_threshold_px, gray_image):
    """
    Merge receipt regions using text pattern analysis and alignment detection.
    """
    if not regions:
        return regions
    
    merged = []
    used = set()
    
    for i, region1 in enumerate(regions):
        if i in used:
            continue
            
        current_region = list(region1)
        used.add(i)
        
        # Keep checking for mergeable regions until no more found
        merged_something = True
        while merged_something:
            merged_something = False
            
            for j, region2 in enumerate(regions):
                if j in used or j == i:
                    continue
                
                if should_merge_regions(current_region, region2, gray_image, proximity_threshold_px):
                    # Merge the regions
                    x1, y1, w1, h1 = current_region
                    x2, y2, w2, h2 = region2
                    current_region = [
                        min(x1, x2),
                        min(y1, y2),
                        max(x1 + w1, x2 + w2) - min(x1, x2),
                        max(y1 + h1, y2 + h2) - min(y1, y2)
                    ]
                    used.add(j)
                    merged_something = True
        
        merged.append(tuple(current_region))
    
    return merged

def cleanup_debug_directory(debug_base_dir, max_dirs_to_keep=5):
    """Clean up old debug directories, keeping only the most recent ones"""
    try:
        # List all debug directories
        debug_dirs = [d for d in os.listdir(debug_base_dir) if d.startswith('debug_')]
        debug_dirs.sort(reverse=True)  # Sort newest to oldest
        
        # Remove old directories beyond the limit
        for old_dir in debug_dirs[max_dirs_to_keep:]:
            old_path = os.path.join(debug_base_dir, old_dir)
            try:
                shutil.rmtree(old_path)
                print(f"Cleaned up old debug directory: {old_dir}")
            except Exception as e:
                print(f"Error cleaning up {old_dir}: {str(e)}")
    except Exception as e:
        print(f"Error during debug cleanup: {str(e)}")

def create_debug_directory(output_folder):
    """Create a timestamped debug directory and clean up old ones"""
    # Create base debug directory if it doesn't exist
    debug_base = os.path.join(output_folder, "debug")
    os.makedirs(debug_base, exist_ok=True)
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = os.path.join(debug_base, f"debug_{timestamp}")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Clean up old debug directories
    cleanup_debug_directory(debug_base)
    
    return debug_dir

def find_text_boundaries(gray_image, x, y, w, h, padding_cm=1.0):
    """
    Find the actual text boundaries within a region and add padding.
    Returns: (new_x, new_y, new_w, new_h)
    """
    # Convert cm to pixels (assuming 96 DPI)
    padding_px = int(padding_cm * 37.8)
    
    # Extract the region of interest
    roi = gray_image[y:y+h, x:x+w]
    
    # Apply threshold to get text
    _, binary = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Find text pixels
    text_pixels = np.where(binary > 0)
    
    if len(text_pixels[0]) == 0:  # No text found
        return x, y, w, h
    
    # Find actual text boundaries
    min_y, max_y = np.min(text_pixels[0]), np.max(text_pixels[0])
    min_x, max_x = np.min(text_pixels[1]), np.max(text_pixels[1])
    
    # Add padding and ensure we don't go outside image bounds
    height, width = gray_image.shape
    new_x = max(0, x + min_x - padding_px)
    new_y = max(0, y + min_y - padding_px)
    new_w = min(width - new_x, (max_x - min_x + 2 * padding_px))
    new_h = min(height - new_y, (max_y - min_y + 2 * padding_px))
    
    return new_x, new_y, new_w, new_h

def detect_receipts(image):
    """Detect and return coordinates of multiple receipts in an image"""
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    
    # Create debug directory with timestamp
    debug_dir = create_debug_directory(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Store original dimensions for relative calculations
    height, width = gray.shape
    min_area = width * height * 0.01  # Minimum 1% of image area
    max_area = width * height * 0.95  # Maximum 95% of image area
    
    # Try multiple threshold approaches
    receipt_contours = []
    
    # 1. Try binary threshold first
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(debug_dir, "1_binary.png"), binary)
    
    # Create kernels for different orientations
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    
    # Process for both orientations
    for kernel in [vertical_kernel, horizontal_kernel]:
        # Dilate to connect components
        dilated = cv2.dilate(binary, kernel, iterations=3)
        cv2.imwrite(os.path.join(debug_dir, f"dilated_{kernel.shape}.png"), dilated)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Accept both vertical and horizontal receipts with wider range
                if 0.1 < aspect_ratio < 10:
                    receipt_contours.append((x, y, w, h))
    
    # 2. If no contours found, try adaptive threshold
    if not receipt_contours:
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 21, 11)
        cv2.imwrite(os.path.join(debug_dir, "2_adaptive.png"), adaptive)
        
        for kernel in [vertical_kernel, horizontal_kernel]:
            dilated = cv2.dilate(adaptive, kernel, iterations=3)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    if 0.1 < aspect_ratio < 10:
                        receipt_contours.append((x, y, w, h))
    
    # 3. If still no contours found, try edge detection
    if not receipt_contours:
        edges = cv2.Canny(gray, 50, 150)
        cv2.imwrite(os.path.join(debug_dir, "3_edges.png"), edges)
        
        for kernel in [vertical_kernel, horizontal_kernel]:
            dilated = cv2.dilate(edges, kernel, iterations=3)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    if 0.1 < aspect_ratio < 10:
                        receipt_contours.append((x, y, w, h))
    
    # Save intermediate debug image
    debug_image = image_np.copy()
    for i, (x, y, w, h) in enumerate(receipt_contours):
        cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(debug_image, f"Initial {i+1}", (x+10, y+30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(debug_dir, "4_initial_detection.png"), 
                cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
    
    # Remove duplicates
    unique_contours = []
    for contour in receipt_contours:
        if not any(regions_overlap(contour, existing, 20) for existing in unique_contours):
            unique_contours.append(contour)
    
    # Merge nearby regions
    proximity_threshold_px = int(37.8)  # 1cm in pixels
    merged_contours = merge_nearby_regions(unique_contours, proximity_threshold_px, gray)
    
    # Apply padding to merged regions
    final_contours = []
    for x, y, w, h in merged_contours:
        # Add padding (1cm on each side)
        padding = int(37.8)  # 1cm in pixels
        new_x = max(0, x - padding)
        new_y = max(0, y - padding)
        new_w = min(width - new_x, w + (2 * padding))
        new_h = min(height - new_y, h + (2 * padding))
        final_contours.append((new_x, new_y, new_w, new_h))
    
    # Sort contours top-to-bottom, then left-to-right
    final_contours.sort(key=lambda x: (x[1], x[0]))
    
    # Save debug images
    debug_image = image_np.copy()
    
    # Draw original contours in blue
    for i, (x, y, w, h) in enumerate(unique_contours):
        cv2.rectangle(debug_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(debug_image, f"Original {i+1}", (x+10, y+30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imwrite(os.path.join(debug_dir, "5_original_contours.png"), 
                cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
    
    # Draw final contours in green
    debug_image = image_np.copy()
    for i, (x, y, w, h) in enumerate(final_contours):
        cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(debug_image, f"Receipt {i+1}", (x+10, y+30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(debug_dir, "6_final_receipts.png"), 
                cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
    
    print(f"Found {len(final_contours)} receipt regions")
    return final_contours

def regions_overlap(region1, region2, threshold):
    """Check if two regions overlap or are close to each other within the threshold"""
    x1_1, y1_1, x2_1, y2_1 = region1
    x1_2, y1_2, x2_2, y2_2 = region2
    
    return not (x2_1 + threshold < x1_2 or x1_1 > x2_2 + threshold or
               y2_1 + threshold < y1_2 or y1_1 > y2_2 + threshold)

def extract_text_from_pdf_page(page, input_path, page_num):
    """Extract text from multiple receipts on a PDF page"""
    texts = []
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            images = convert_from_path(
                pdf_path=input_path,
                first_page=page_num + 1,
                last_page=page_num + 1,
                poppler_path=r'C:\Program Files\poppler-24.08.0\Library\bin'
            )
            
            if images:
                # Detect multiple receipts
                receipt_regions = detect_receipts(images[0])
                
                for x, y, w, h in receipt_regions:
                    # Crop the receipt region
                    receipt_image = images[0].crop((x, y, x+w, y+h))
                    # Extract text from the receipt
                    text = pytesseract.image_to_string(receipt_image)
                    texts.append(text)
                
    except Exception as e:
        print(f"OCR Error: {str(e)}")
    
    return texts

def extract_date(text):
    """Extract date from receipt text in YYMMDD format"""
    # Common date patterns (add more as needed)
    date_patterns = [
        r'\d{2}/\d{2}/\d{2}',  # MM/DD/YY
        r'\d{2}-\d{2}-\d{2}',  # MM-DD-YY
        r'\d{2}\.\d{2}\.\d{2}'  # MM.DD.YY
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            date_str = match.group(0)
            try:
                # Convert to datetime object
                date = datetime.strptime(date_str, "%m/%d/%y")
                # Return in YYMMDD format
                return date.strftime("%y%m%d")
            except ValueError:
                continue
    return "000000"  # Default if no date found

def extract_vendor(text):
    """Extract vendor name from receipt text"""
    # Common store names to check first
    common_stores = [
        (r'HOME\s*DEPOT', 'HOME_DEPOT'),
        (r'COSTCO\s*WHOLESALE', 'COSTCO'),
        (r'WALMART', 'WALMART'),
        (r'TARGET', 'TARGET')
    ]
    
    # Check for common store names first
    for pattern, store_name in common_stores:
        if re.search(pattern, text, re.IGNORECASE):
            return store_name
    
    # Look for common vendor indicators
    vendor_patterns = [
        r'STORE:[\s]*([^\n]+)',
        r'MERCHANT:[\s]*([^\n]+)',
        r'VENDOR:[\s]*([^\n]+)',
        # Add more patterns as needed
    ]
    
    for pattern in vendor_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If no pattern matches, try to get the first line
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        # Try to find the first meaningful line that might be a store name
        for line in lines:
            # Skip lines that are likely not store names
            if re.search(r'^\d|TOTAL|DATE|TIME|RECEIPT|#\d+', line, re.IGNORECASE):
                continue
            return line[:30]  # Limit length
    
    return "UNKNOWN_VENDOR"

def extract_amount(text):
    """Extract total amount from receipt text"""
    # Split text into lines and remove empty lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Look for total amount patterns, prioritizing ones that typically indicate final total
    final_total_patterns = [
        (r'TOTAL\s+AMOUNT\s*[\.:]?\s*\$?\s*(\d+\.\d{2})', 10),
        (r'GRAND\s+TOTAL\s*[\.:]?\s*\$?\s*(\d+\.\d{2})', 9),
        (r'BALANCE\s+DUE\s*[\.:]?\s*\$?\s*(\d+\.\d{2})', 8),
        (r'TOTAL\s+DUE\s*[\.:]?\s*\$?\s*(\d+\.\d{2})', 7),
        (r'ORDER\s+TOTAL\s*[\.:]?\s*\$?\s*(\d+\.\d{2})', 6),
        (r'TOTAL\s*[\.:]?\s*\$?\s*(\d+\.\d{2})', 5)
    ]
    
    # Store all found amounts with their line numbers and pattern priority
    found_amounts = []
    
    for line_num, line in enumerate(lines):
        for pattern, priority in final_total_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                amount = float(match.group(1))
                # Calculate score based on position in receipt and pattern priority
                position_score = line_num / len(lines)  # Higher score for amounts near the end
                score = position_score * 0.7 + (priority / 10) * 0.3  # Weight position more than pattern
                found_amounts.append((amount, score))
    
    if found_amounts:
        # Sort by score and take the highest scoring amount
        found_amounts.sort(key=lambda x: x[1], reverse=True)
        return f"{found_amounts[0][0]:.2f}"
    
    return "0.00"

def format_amount(amount_str):
    """Format amount with dollar sign and commas"""
    try:
        # Convert string to float
        amount = float(amount_str)
        # Format with dollar sign and commas
        return f"${amount:,.2f}"
    except ValueError:
        return f"${0:,.2f}"

def create_filename(date, vendor, amount):
    """Create a clean filename from receipt information"""
    # Clean vendor name (remove special characters)
    vendor = re.sub(r'[^a-zA-Z0-9\s]', '', vendor)
    vendor = vendor.strip().replace(' ', '_')
    
    # Format the amount with dollar sign and commas
    formatted_amount = format_amount(amount)
    
    return f"{date}_{vendor}_{formatted_amount}.pdf"

def split_pages(input_path, output_folder):
    """Split PDF pages into separate files"""
    try:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Create debug folder
        debug_dir = os.path.join(output_folder, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Open the PDF file
        pdf = PdfReader(input_path)
        print(f"Successfully opened PDF with {len(pdf.pages)} pages")
        
        # Process each page
        for page_num in range(len(pdf.pages)):
            print(f"\nProcessing page {page_num + 1}")
            
            # Convert page to image
            images = convert_from_path(
                input_path,
                first_page=page_num + 1,
                last_page=page_num + 1,
                poppler_path=r'C:\Program Files\poppler-24.08.0\Library\bin'
            )
            
            if not images:
                print(f"No image found for page {page_num + 1}")
                continue
                
            page_image = images[0]
            
            # Save original page image for reference
            page_image.save(os.path.join(debug_dir, f"page_{page_num + 1}_original.png"))
            
            # Detect receipt regions
            receipt_regions = detect_receipts(page_image)
            print(f"Found {len(receipt_regions)} receipt regions on page {page_num + 1}")
            
            # Process each receipt region
            for i, (x, y, w, h) in enumerate(receipt_regions):
                try:
                    # Crop the image to get just this receipt
                    receipt_image = page_image.crop((x, y, x+w, y+h))
                    
                    # Save debug image of cropped receipt
                    receipt_image.save(os.path.join(debug_dir, f"page_{page_num + 1}_receipt_{i + 1}.png"))
                    
                    # Extract text from this specific receipt region
                    text = pytesseract.image_to_string(receipt_image)
                    print(f"\nProcessing receipt {i+1}:")
                    print(f"Extracted text length: {len(text)}")
                    
                    # Save extracted text for debugging
                    with open(os.path.join(debug_dir, f"page_{page_num + 1}_receipt_{i + 1}_text.txt"), 'w') as f:
                        f.write(text)
                    
                    # Extract receipt information
                    date = extract_date(text)
                    vendor = extract_vendor(text)
                    amount = extract_amount(text)
                    
                    print(f"Extracted info - Date: {date}, Vendor: {vendor}, Amount: {amount}")
                    
                    # Create filename
                    base_filename = create_filename(date, vendor, amount)
                    # Always add receipt number if multiple receipts found
                    if len(receipt_regions) > 1:
                        filename = f"{base_filename[:-4]}_{i+1}.pdf"
                    else:
                        filename = base_filename
                    
                    # Save the cropped image temporarily
                    temp_image_path = os.path.join(output_folder, f"temp_receipt_{i}.png")
                    receipt_image.save(temp_image_path)
                    
                    # Create a new PDF from the cropped image
                    pdf_writer = PdfWriter()
                    with Image.open(temp_image_path) as img:
                        # Convert to RGB if needed
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        # Create PDF from image
                        img_path = os.path.join(output_folder, f"temp_receipt_{i}.pdf")
                        img.save(img_path, 'PDF', resolution=100.0)
                        
                        # Read the temporary PDF and add it to the writer
                        temp_pdf = PdfReader(img_path)
                        pdf_writer.add_page(temp_pdf.pages[0])
                    
                    # Save the final PDF
                    output_filename = os.path.join(output_folder, filename)
                    with open(output_filename, 'wb') as output_file:
                        pdf_writer.write(output_file)
                    
                    # Clean up temporary files
                    os.remove(temp_image_path)
                    os.remove(img_path)
                    
                    print(f'Created: {output_filename}')
                
                except Exception as e:
                    print(f"Error processing receipt {i+1}: {str(e)}")
                    continue
                
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise

def cluster_text_regions(image):
    """Identify text clusters using connected components analysis"""
    # Get the debug directory from the parent directory structure
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    debug_base = os.path.join(base_dir, "debug")
    debug_dirs = sorted(os.listdir(debug_base))
    debug_dir = os.path.join(debug_base, debug_dirs[-1]) if debug_dirs else debug_base
    
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Multi-stage thresholding to catch all text
    binary_layers = []
    thresholds = [
        (180, cv2.THRESH_BINARY_INV),  # Dark text
        (120, cv2.THRESH_BINARY_INV),  # Medium text
        (200, cv2.THRESH_BINARY)       # Light background noise
    ]
    
    for threshold, method in thresholds:
        _, binary = cv2.threshold(gray, threshold, 255, method)
        binary_layers.append(binary)
        
        # Save debug image
        cv2.imwrite(os.path.join(debug_dir, f"binary_threshold_{threshold}.png"), binary)
    
    # Combine binary layers
    combined_binary = cv2.bitwise_or(binary_layers[0], binary_layers[1])
    combined_binary = cv2.bitwise_and(combined_binary, cv2.bitwise_not(binary_layers[2]))
    
    # Save combined binary
    cv2.imwrite(os.path.join(debug_dir, "combined_binary.png"), combined_binary)
    
    # Create kernels for text connection
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # Connect horizontal text
    
    # Connect nearby text while preserving gaps
    connected = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, connect_kernel)
    connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, text_kernel)
    
    # Save connected components debug image
    cv2.imwrite(os.path.join(debug_dir, "connected_text.png"), connected)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(connected)
    
    # Filter and merge text clusters
    height, width = gray.shape
    min_size = width * height * 0.001  # Minimum 0.1% of image area
    min_width = width * 0.05  # Minimum 5% of image width
    min_height = height * 0.05  # Minimum 5% of image height
    text_clusters = []
    
    for i in range(1, num_labels):  # Skip background (label 0)
        x, y, w, h, area = stats[i]
        
        # Calculate text density in region
        roi = combined_binary[y:y+h, x:x+w]
        text_pixels = np.sum(roi == 255)
        density = text_pixels / (w * h)
        
        # Filter based on size and density
        if (area > min_size and w > min_width and h > min_height and 
            0.05 < density < 0.5):  # Adjusted density range
            text_clusters.append((x, y, w, h))
    
    return text_clusters

def validate_spacing(clusters, min_space_cm=1.0):
    """Validate and merge clusters based on spacing"""
    min_space_px = int(min_space_cm * 37.8)  # Convert cm to pixels
    merged_clusters = []
    used_clusters = set()
    
    # Sort clusters by position (top to bottom, left to right)
    clusters = sorted(clusters, key=lambda x: (x[1], x[0]))
    
    for i, cluster1 in enumerate(clusters):
        if i in used_clusters:
            continue
            
        current_cluster = list(cluster1)
        used_clusters.add(i)
        merged = True
        
        while merged:
            merged = False
            for j, cluster2 in enumerate(clusters):
                if j in used_clusters:
                    continue
                    
                # Calculate distances
                x1, y1, w1, h1 = current_cluster
                x2, y2, w2, h2 = cluster2
                
                # Check if clusters are part of the same receipt
                horizontal_overlap = (x1 < (x2 + w2) and (x1 + w1) > x2)
                vertical_overlap = (y1 < (y2 + h2) and (y1 + h1) > y2)
                
                # Calculate minimum distance between clusters
                if horizontal_overlap:
                    distance = min(abs(y1 - (y2 + h2)), abs((y1 + h1) - y2))
                elif vertical_overlap:
                    distance = min(abs(x1 - (x2 + w2)), abs((x1 + w1) - x2))
                else:
                    continue  # Skip if no overlap in either direction
                
                # Merge if distance is less than minimum space
                if distance < min_space_px:
                    current_cluster = [
                        min(x1, x2),
                        min(y1, y2),
                        max(x1 + w1, x2 + w2) - min(x1, x2),
                        max(y1 + h1, y2 + h2) - min(y1, y2)
                    ]
                    used_clusters.add(j)
                    merged = True
                    break
        
        merged_clusters.append(tuple(current_cluster))
    
    return merged_clusters

def form_receipt_rectangles(clusters):
    """Form final receipt rectangles from validated clusters"""
    if not clusters:
        return []
    
    receipt_regions = []
    
    for x, y, w, h in clusters:
        # Add padding around the cluster (0.5cm on each side)
        padding = int(0.5 * 37.8)  # 0.5cm in pixels
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = w + (2 * padding)
        h = h + (2 * padding)
        
        # Ensure aspect ratio is reasonable for a receipt
        aspect_ratio = w / h if h > 0 else 0
        
        # Accept both vertical (0.2-0.8) and horizontal (1.2-5.0) receipts
        if (0.2 < aspect_ratio < 0.8) or (1.2 < aspect_ratio < 5.0):
            receipt_regions.append((x, y, w, h))
    
    return receipt_regions

if __name__ == "__main__":
    # Use absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_pdf = os.path.join(script_dir, "..", "input", "document.pdf")
    output_dir = os.path.join(script_dir, "..", "output")
    
    print(f"Input PDF path: {input_pdf}")
    print(f"Output directory: {output_dir}")
    print(f"Input file exists: {os.path.exists(input_pdf)}")
    
    split_pages(input_pdf, output_dir)