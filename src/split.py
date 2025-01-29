# split.py
from PyPDF2 import PdfReader, PdfWriter  # Updated import statement
import os

# Add debugging information
print("Script starting...")
print("Current working directory:", os.getcwd())

def split_pages(input_path, output_folder):
    print(f"Attempting to process: {input_path}")
    print(f"Output folder: {output_folder}")
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return
    
    # Read the PDF file
    try:
        pdf = PdfReader(input_path)  # Updated to PdfReader
        print(f"Successfully opened PDF with {len(pdf.pages)} pages")
        
        # Process each page
        for page_num in range(len(pdf.pages)):
            # Create PDF writer object
            pdf_writer = PdfWriter()  # Updated to PdfWriter
            # Add the current page
            pdf_writer.add_page(pdf.pages[page_num])
            
            # Create output filename
            output_filename = os.path.join(
                output_folder, 
                f'page_{page_num + 1}.pdf'
            )
            
            # Write the page to a new PDF
            with open(output_filename, 'wb') as output_file:
                pdf_writer.write(output_file)
            
            print(f'Created: {output_filename}')
            
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Use absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_pdf = os.path.join(script_dir, "input", "document.pdf")
    output_dir = os.path.join(script_dir, "output")
    
    print(f"Input PDF path: {input_pdf}")
    print(f"Output directory: {output_dir}")
    print(f"Input file exists: {os.path.exists(input_pdf)}")
    
    split_pages(input_pdf, output_dir)