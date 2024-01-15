# Generate Basin Boundary Fractals with Orbits

This project focuses on the generation of basin boundary fractals using specific orbital data. It includes tools for creating detailed plots and videos to visualize these fractals.

## Project Structure

- `code/`
  - `example_figure_eight.ipynb`: Notebook for generating fractal data based on the figure-eight orbit.
  - `example_generate_plots.ipynb`: Notebook for creating plots and compiling them into a video after data generation.
  - `plot_utils.py`: Utility functions dedicated to plotting.
  - `utils.py`: Utility functions for orbit integration and fractal generation.

- `data/`
  - `periodic_3b_inits.json`: JSON file containing initial conditions for some periodic three-body orbits.

## Requirements

This project uses the following Python packages:
- `numpy`: For numerical operations.
- `numba`: To accelerate Python functions.
- `matplotlib`: For creating plots and visualizations.

## Getting Started

1. Clone the repository to your local machine.
2. Install the required packages using `pip install numpy numba matplotlib`.
3. Navigate to the `code/` directory.
4. Run `example_figure_eight.ipynb` to generate the fractal data.
5. Execute `example_generate_plots.ipynb` to create plots and a video from the generated data.

## Contributing

Contributions to the project are welcome, specially regarding efficiency of the code. If you have suggestions or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

