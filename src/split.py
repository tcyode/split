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

# Add debugging information
print("Script starting...")
print("Current working directory:", os.getcwd())

# Set Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_receipts(image):
    """Detect and return coordinates of multiple receipts in an image"""
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Apply binary thresholding with a lower threshold
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # Create a kernel for morphological operations
    kernel = np.ones((15,15), np.uint8)  # Increased kernel size
    
    # Dilate to connect components within receipts
    dilated = cv2.dilate(binary, kernel, iterations=3)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size and shape
    receipt_contours = []
    min_area = image.width * image.height * 0.01  # Reduced to 1% of image area
    max_area = image.width * image.height * 0.95  # Maximum 95% of image area
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Accept both vertical (aspect_ratio < 1) and horizontal (aspect_ratio > 1) receipts
            if 0.1 < aspect_ratio < 10:  # Widened aspect ratio range
                receipt_contours.append((x, y, w, h))
    
    # If no contours found with binary threshold, try adaptive threshold
    if not receipt_contours:
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 21, 11)  # Adjusted block size and C
        dilated = cv2.dilate(adaptive, kernel, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                if 0.1 < aspect_ratio < 10:
                    receipt_contours.append((x, y, w, h))
    
    # If still no contours found, try edge detection
    if not receipt_contours:
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        dilated = cv2.dilate(edges, kernel, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                if 0.1 < aspect_ratio < 10:
                    receipt_contours.append((x, y, w, h))
    
    # Sort contours left-to-right, then top-to-bottom
    receipt_contours.sort(key=lambda x: (x[1], x[0]))
    
    # Draw debug image
    debug_dir = "debug_images"
    os.makedirs(debug_dir, exist_ok=True)
    debug_image = image_np.copy()
    for i, (x, y, w, h) in enumerate(receipt_contours):
        cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Add region number
        cv2.putText(debug_image, str(i+1), (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imwrite(os.path.join(debug_dir, f"detected_regions_{len(receipt_contours)}.png"), cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
    
    # Also save the intermediate processing images for debugging
    cv2.imwrite(os.path.join(debug_dir, "binary.png"), binary)
    cv2.imwrite(os.path.join(debug_dir, "dilated.png"), dilated)
    
    print(f"Found {len(receipt_contours)} receipt regions")
    return receipt_contours

def regions_overlap(region1, region2, threshold):
    """Check if two regions overlap or are close to each other"""
    x1_1, y1_1, x2_1, y2_1 = region1
    x1_2, y1_2, x2_2, y2_2 = region2
    
    return not (x2_1 + threshold < x1_2 or x1_1 > x2_2 + threshold or
               y2_1 + threshold < y1_2 or y1_1 > y2_2 + threshold)

def merge_regions(region1, region2):
    """Merge two regions into one with improved overlap detection"""
    x1_1, y1_1, x2_1, y2_1 = region1
    x1_2, y1_2, x2_2, y2_2 = region2
    
    # Calculate area of overlap
    overlap_x1 = max(x1_1, x1_2)
    overlap_y1 = max(y1_1, y1_2)
    overlap_x2 = min(x2_1, x2_2)
    overlap_y2 = min(y2_1, y2_2)
    
    if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
        # There is overlap, merge the regions
        return [
            min(x1_1, x1_2),
            min(y1_1, y1_2),
            max(x2_1, x2_2),
            max(y2_1, y2_2)
        ]
    else:
        # No overlap, check if regions are close
        distance_x = min(abs(x1_1 - x2_2), abs(x2_1 - x1_2))
        distance_y = min(abs(y1_1 - y2_2), abs(y2_1 - y1_2))
        
        if distance_x < 100 and distance_y < 100:  # Adjust threshold as needed
            return [
                min(x1_1, x1_2),
                min(y1_1, y1_2),
                max(x2_1, x2_2),
                max(y2_1, y2_2)
            ]
        return None

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

if __name__ == "__main__":
    # Use absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_pdf = os.path.join(script_dir, "..", "input", "document.pdf")
    output_dir = os.path.join(script_dir, "..", "output")
    
    print(f"Input PDF path: {input_pdf}")
    print(f"Output directory: {output_dir}")
    print(f"Input file exists: {os.path.exists(input_pdf)}")
    
    split_pages(input_pdf, output_dir)