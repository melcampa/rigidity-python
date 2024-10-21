import pandas as pd
import matplotlib.pyplot as plt
import glob

# Create a list to hold DataFrames for each CSV file
dataframes = []

# Read all CSV files from the specified directory
csv_files = glob.glob('C:/Users/chipr/OneDrive/Desktop/sim/rigidity-python/results/*.csv')

# Check if CSV files are found
if not csv_files:
    print("No CSV files found in the specified directory.")
else:
    print(f"Found {len(csv_files)} CSV files.")

# Load each CSV file into a DataFrame
for file in csv_files:
    df = pd.read_csv(file)
    if 'Iteration' in df.columns and 'Average Distance' in df.columns:
        dataframes.append(df)
    else:
        print(f"Warning: '{file}' does not have the required columns.")

# Create the plot
plt.figure(figsize=(10, 6))

# Plot each DataFrame as a scatter plot
for i, df in enumerate(dataframes):
    # Sample the data to plot only every 10th point
    sampled_df = df[df['Iteration'] % 10 == 0]
    plt.scatter(sampled_df['Iteration'], sampled_df['Average Distance'], label=f'Experiment {i + 1}', alpha=0.6)

    #print(f"Plotted data from Experiment {i + 1} with {len(df)} data points.")

# Set plot labels and title
plt.xlabel('Iteration')
plt.ylabel('Average Distance')
plt.title('Average Distance Over Iterations for Multiple Experiments')
plt.xlim(-10, 99)  # Set x-axis limits based on the number of iterations
plt.ylim(4, 13)  # Set y-axis limits based on the maximum average distance
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
