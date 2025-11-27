# Instructions to Generate Visualization Images

## Quick Steps

1. **Activate your Python environment** (if you're using one):
   ```bash
   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

2. **Ensure all dependencies are installed**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the image generation script**:
   ```bash
   python scripts/generate_images.py
   ```

4. **Check the images folder**:
   The script will create 5 PNG files in the `images/` folder:
   - `01_data_availability_heatmap.png`
   - `02_countries_per_year.png`
   - `03_correlation_heatmap.png`
   - `04_elbow_silhouette.png`
   - `05_cluster_distributions.png`

## If You Encounter Errors

### Missing Module Error
If you see `ModuleNotFoundError`, ensure you've installed all requirements:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly scipy
```

### File Not Found Error
Make sure you're running the script from the project root directory (where README.md is located).

## Alternative: Extract Images from Notebooks

If the script doesn't work, you can manually export images from the notebooks:

1. Open the relevant notebook in Jupyter
2. Run all cells
3. Right-click on the visualization
4. Select "Save Image As..."
5. Save to the `images/` folder with the appropriate name

## Image Specifications

All images are generated at **300 DPI** for high quality display in the README and presentations.

## Troubleshooting

**Problem:** Script runs but creates empty/small images
**Solution:** Check that the data files exist in `data/processing/` folder

**Problem:** Color/style doesn't match what you want
**Solution:** Edit `scripts/generate_images.py` and adjust the matplotlib/seaborn styling parameters

**Problem:** Want to regenerate just one image
**Solution:** Comment out the sections in `scripts/generate_images.py` for the images you don't need to regenerate
