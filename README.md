# Airfoil Analysis with NeuralFoil and XFOIL

This script provides tools for analyzing airfoils using **NeuralFoil** and **XFOIL**. It allows for the evaluation of multiple airfoils by calculating aerodynamic properties such as lift coefficient (CL), drag coefficient (CD), and moment coefficient (CM) at specified Reynolds numbers. The script can analyze a batch of airfoils, compute their maximum CL, and rank them based on a scoring function.

## Features

- Analyze airfoils using **NeuralFoil** or **XFOIL**
- Compute aerodynamic coefficients across a range of Reynolds numbers
- Determine **maximum CL** for a given airfoil
- Score airfoils based on aerodynamic efficiency
- Support for batch airfoil processing
- Generate plots for comparison
- Save results as `.csv` files for further analysis

## Installation

To use this script, install the required dependencies by running:

```sh
pip install -r requirements.txt
```

## Dependencies

This script relies on the following Python libraries:

- `aerosandbox`
- `numpy`
- `pandas`
- `matplotlib`
- `alive-progress` (for optional progress bars)

## Usage

### Running an Airfoil Batch Analysis

The script allows batch processing of airfoils stored in `.dat` files:

```python
from airfoil_analysis import BatchAirfoil

airfoil_database_path = "./coord_seligFmt"  # Path to airfoil files
maxCL_reynolds = 1.225 * 10 * 0.5 / (1.7894e-5)  # Example Reynolds number
CL_selection = [0.1, 0.3]  # CL range for analysis
Reynolds = [1239401]  # Reynolds numbers to analyze

# Initialize batch airfoil analysis
batch = BatchAirfoil(airfoil_database_path, CL_selection, Reynolds, maxCL_reynolds=maxCL_reynolds)

# Run analysis
batch.run_batch()

# Plot and save results
batch.draw_analysis(save=True, topN=5)
batch.save_results(topN=20, filename="top20_airfoil.csv")
batch.save_results(topN=None, filename="full_airfoil.csv")
```

### Comparing Top Airfoils

To visualize and compare the performance of selected airfoils:

```python
from airfoil_analysis import compare
import aerosandbox as asb

top_airfoils = ['sd7032', 'hq2090sm', 'rg12a', 's2048']
airfoils = [asb.Airfoil(f) for f in top_airfoils]

# Compare selected airfoils
compare(airfoils=airfoils, reynolds=1239401, save=False)

# Visualize the airfoil shapes
for af in airfoils:
    af.draw()
```

## Notes

- **XFOIL Support:** The script currently does not support running XFOIL (function `run_xfoil` is not implemented). Only **NeuralFoil** is functional.
- **Airfoil Data:** The script expects airfoil coordinate files in `.dat` format stored in a specified directory.
- **NeuralFoil Confidence Threshold:** The script rejects results with an analysis confidence below 0.9.
- **Scoring:** The airfoils are ranked based on their aerodynamic performance using a weighted scoring function.

## License

This project is licensed under the MIT License.
