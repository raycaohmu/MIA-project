# Data Directory

This directory contains the input data for the MIA project. Due to file size limitations, the actual data files are not included in the Git repository.

## Required Structure

```
data/
├── README.md                  # This file
├── slide_data.csv            # Sample labels (create your own)
├── slide_ov_response.csv     # Ovarian cancer response data (create your own)
├── wsi/                      # Place your WSI files here
│   ├── sample1.svs
│   ├── sample2.svs
│   └── ...
└── wsi_output/              # Output from processing (auto-generated)
    ├── sampled_sample1/
    ├── sampled_sample2/
    └── ...
```

## Data Preparation

### 1. WSI Files
Place your Whole Slide Image files (.svs, .tif, .tiff, .ndpi, .mrxs) in the `wsi/` directory.

### 2. Label Files
Create CSV files with the following format:

**slide_data.csv / slide_ov_response.csv:**
```csv
filename,label
sample1,0
sample2,1
sample3,0
```

Where:
- `filename`: WSI filename without extension
- `label`: Classification label (0 or 1 for binary classification)

## Data Sources

- WSI files should be obtained from your institution or public datasets
- Ensure proper data permissions and privacy compliance
- Common public datasets:
  - TCGA (The Cancer Genome Atlas)
  - CAMELYON series
  - Institution-specific datasets

## File Size Considerations

- WSI files are typically 500MB - 5GB each
- Processing outputs can be 100MB - 1GB per WSI
- Ensure sufficient storage space (10GB+ recommended)

## Security Notes

- Never commit patient data to version control
- Use data encryption for sensitive medical data
- Follow institutional data handling policies
- Consider using Git LFS for large files if needed
