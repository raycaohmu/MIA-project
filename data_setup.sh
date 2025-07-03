#!/bin/bash

# Data Setup Script for MIA Project
# This script helps you prepare the data directory structure

echo "📁 MIA Data Setup"
echo "=================="

# Create data directory structure
echo "Creating data directory structure..."
mkdir -p data/wsi
mkdir -p data/wsi_output
mkdir -p output/features
mkdir -p output/graph_output
mkdir -p output/models
mkdir -p logs

echo "✅ Directory structure created"

# Check for WSI files
wsi_count=$(find data/wsi -name "*.svs" -o -name "*.tif" -o -name "*.tiff" -o -name "*.ndpi" -o -name "*.mrxs" 2>/dev/null | wc -l)

if [ $wsi_count -eq 0 ]; then
    echo "⚠️  No WSI files found in data/wsi/"
    echo "   Please add your WSI files to data/wsi/ directory"
    echo "   Supported formats: .svs, .tif, .tiff, .ndpi, .mrxs"
else
    echo "✅ Found $wsi_count WSI files"
fi

# Check for label files
if [ -f "data/slide_ov_response.csv" ] || [ -f "data/slide_data.csv" ]; then
    echo "✅ Label file found"
else
    echo "⚠️  No label file found"
    echo "   Please create a CSV file based on data/slide_data_template.csv"
    echo "   Example:"
    echo "   cp data/slide_data_template.csv data/slide_ov_response.csv"
    echo "   # Then edit the file with your actual data"
fi

# Check disk space
available_space=$(df -h . | awk 'NR==2 {print $4}')
echo "💾 Available disk space: $available_space"

if [ "${available_space%G*}" -lt 20 ] 2>/dev/null; then
    echo "⚠️  Warning: Less than 20GB available space"
    echo "   Consider freeing up space for optimal performance"
fi

echo ""
echo "📋 Next Steps:"
echo "1. Add WSI files to data/wsi/"
echo "2. Create label CSV file (use data/slide_data_template.csv as template)"
echo "3. Download NuLite pretrained model to models/pretrained/NuLite/"
echo "4. Run: bash random_sample_nuclei_detection.sh"
echo ""
echo "📖 For more information, see data/README.md and output/README.md"
