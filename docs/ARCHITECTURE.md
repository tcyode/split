# Receipt Scanner Architecture

## Processing Flow

```mermaid

graph TD
A[PDF Input] --> B[Page Processing]
B --> C[Receipt Detection]
C --> D[Region Analysis]
D --> E[Text Extraction]
E --> F[PDF Output]
subgraph Detection Flow
C --> C1[Threshold]
C1 --> C2[Contour Detection]
C2 --> C3[Region Merging]
end
subgraph Analysis Steps
D --> D1[Text Pattern]
D1 --> D2[Density Check]
D2 --> D3[Rotation Analysis]
end
subgraph Text Processing
E --> E1[OCR]
E1 --> E2[Date Detection]
E1 --> E3[Vendor Detection]
E1 --> E4[Amount Detection]
end
style A fill:#e6f3ff,stroke:#333
style B fill:#fff3e6,stroke:#333
style C fill:#ffe6e6,stroke:#333
style D fill:#e6ffe6,stroke:#333
style E fill:#f9f9f9,stroke:#333
style F fill:#e6f3ff,stroke:#333


## Code Structure

mermaid
graph LR
A[split.py] --> B[Detection]
A --> C[Analysis]
A --> D[Extraction]
subgraph Detection Functions
B --> B1[detect_receipts]
B --> B2[merge_nearby_regions]
B --> B3[correct_skew]
end
subgraph Analysis Functions
C --> C1[analyze_text_pattern]
C --> C2[evaluate_rotation]
C --> C3[measure_text_alignment]
end
subgraph Extraction Functions
D --> D1[extract_date]
D --> D2[extract_vendor]
D --> D3[extract_amount]
end
style A fill:#e6f3ff,stroke:#333
style B fill:#fff3e6,stroke:#333
style C fill:#ffe6e6,stroke:#333
style D fill:#e6ffe6,stroke:#333


## Debug Visualization Flow

mermaid
graph TD
A[Original Image] --> B[Preprocessing]
B --> C[Detection Steps]
C --> D[Final Output]
subgraph Debug Images
B --> B1[Grayscale]
B --> B2[Threshold]
B --> B3[Edge Detection]
C --> C1[Initial Contours]
C --> C2[Merged Regions]
C --> C3[Rotation Correction]
D --> D1[Extracted Receipts]
D --> D2[Text Regions]
D --> D3[Final PDFs]
end
style A fill:#f9f9f9,stroke:#333
style B fill:#e6f3ff,stroke:#333
style C fill:#ffe6e6,stroke:#333
style D fill:#e6ffe6,stroke:#333

## Key Components

### Receipt Detection
- Threshold-based detection
- Contour analysis
- Region merging
- Rotation correction

### Text Analysis
- Pattern recognition
- Density validation
- Aspect ratio checking
- Alignment detection

### Data Extraction
- OCR processing
- Date parsing
- Vendor identification
- Amount detection

### Debug Visualization
- Initial detection
- Merged regions
- Rotation correction
- Final output