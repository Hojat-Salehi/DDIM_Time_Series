import pandas as pd
import numpy as np
from datetime import datetime, timedelta

df = pd.read_csv('data/master_data_IBM.csv')

# transform time to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Get all unique dates from your data
unique_dates = df['timestamp'].dt.date.unique()
print("Step 1 - Unique dates:")
print(unique_dates[:5])  # Show first 5 dates
print("\n")

# Create complete time index for all dates
complete_times = []
for date in unique_dates:
    day_times = pd.date_range(
        start=f"{date} 04:00:00",
        end=f"{date} 19:30:00",
        freq="30min"
    )
    complete_times.extend(day_times)

print("Step 2 - Complete times array first few elements:")
print(complete_times[:5])
print("\n")

# Create a DataFrame with all possible timestamps
complete_df = pd.DataFrame({'timestamp': complete_times})
print("Step 3 - Complete DataFrame head:")
print(complete_df.head())
print("\n")

# Merge with your actual data
merged_df = pd.merge(complete_df, df, on='timestamp', how='left')
print("Step 4 - Merged DataFrame head:")
print(merged_df.head())
print("\n")

print("Step 5 - Merged DataFrame columns:")
print(merged_df.columns)
print("\n")

# Create date column BEFORE groupby
print("Step 6 - Creating date column")
merged_df['date'] = merged_df['timestamp'].dt.date
print("After adding date column, DataFrame columns:")
print(merged_df.columns)
print("\n")
print("First few rows with date column:")
print(merged_df.head())
print("\n")

# Create date column
merged_df['date'] = merged_df['timestamp'].dt.date

# Do both forward and backward fill in one group operation
merged_df = merged_df.groupby('date').apply(lambda x: x.ffill().bfill())

# Drop the helper date column
merged_df = merged_df.drop('date', axis=1)


# Step 1: Let's verify our data structure and check one day's data
test_date = merged_df['timestamp'].dt.date.unique()[0]
print(f"Testing with date: {test_date}")

# Filter data for our test date
day_data = merged_df[merged_df['timestamp'].dt.date == test_date].copy()
print("\nShape of single day data:", day_data.shape)
print("\nColumns available:", day_data.columns.tolist())

# Check time range
print("\nTime range:")
print("Start:", day_data['timestamp'].min())
print("End:", day_data['timestamp'].max())


# Step 1: Calculate statistics for price changes and volume
all_changes = []
increases = []
decreases = []
all_volumes = []

for date in merged_df['timestamp'].dt.date.unique():
    day_data = merged_df[merged_df['timestamp'].dt.date == date].copy()
    
    # Calculate price changes
    first_price = day_data['close'].iloc[0]
    changes = ((day_data['close'] - first_price) / first_price) * 100
    
    # Separate increases and decreases
    increases.extend([c for c in changes if c > 0])
    decreases.extend([c for c in changes if c < 0])
    all_changes.extend(changes)
    
    # Collect volume data
    all_volumes.extend(day_data['volume'])

# Calculate limits for price changes
increase_mean = np.mean(increases)
increase_std = np.std(increases)
decrease_mean = np.mean(decreases)
decrease_std = np.std(decreases)

y_max = increase_mean + 2 * increase_std
y_min = decrease_mean - 2 * decrease_std

# Calculate volume limits
volume_mean = np.mean(all_volumes)
volume_std = np.std(all_volumes)
volume_max = volume_mean + 2 * volume_std

print("Calculated Limits:")
print(f"Price increase limit: {y_max:.2f}%")
print(f"Price decrease limit: {y_min:.2f}%")
print(f"Volume upper limit: {volume_max:.0f}")


# Now create separate plots for one test day
test_date = merged_df['timestamp'].dt.date.unique()[0]
day_data = merged_df[merged_df['timestamp'].dt.date == test_date].copy()

# Calculate price changes for test day
first_price = day_data['close'].iloc[0]
day_data['price_change'] = ((day_data['close'] - first_price) / first_price) * 100

# Split at cutoff
cutoff_time = pd.Timestamp(f"{test_date} 15:30:00")
before_cutoff = day_data[day_data['timestamp'] <= cutoff_time]
after_cutoff = day_data[day_data['timestamp'] >= cutoff_time]

# Set common x-axis limits
x_min = pd.Timestamp(f"{test_date} 04:00:00")
x_max = pd.Timestamp(f"{test_date} 19:30:00")

# Create separate figures for each channel
# Channel 1: Volume
plt.figure(figsize=(10, 5))
plt.plot(before_cutoff['timestamp'], before_cutoff['volume'], 
         color='blue', linewidth=1)
plt.axvline(x=cutoff_time, color='red', linestyle='--', linewidth=1)
plt.ylim(0, volume_max)
plt.xlim(x_min, x_max)
plt.title('Channel 1: Volume')
plt.xticks([])
plt.yticks([])
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.show()

# Channel 2: Price changes before cutoff
plt.figure(figsize=(10, 5))
plt.plot(before_cutoff['timestamp'], before_cutoff['price_change'], 
         color='blue', linewidth=1)
plt.axvline(x=cutoff_time, color='red', linestyle='--', linewidth=1)
plt.ylim(y_min, y_max)
plt.xlim(x_min, x_max)
plt.title('Channel 2: Price Changes Before Cutoff')
plt.xticks([])
plt.yticks([])
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.show()

# Channel 3: Price changes after cutoff
plt.figure(figsize=(10, 5))
plt.plot(after_cutoff['timestamp'], after_cutoff['price_change'], 
         color='blue', linewidth=1)
plt.axvline(x=cutoff_time, color='red', linestyle='--', linewidth=1)
plt.ylim(y_min, y_max)
plt.xlim(x_min, x_max)
plt.title('Channel 3: Price Changes After Cutoff')
plt.xticks([])
plt.yticks([])
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.show()

print("\nPlot settings used:")
print(f"Time range: {x_min} to {x_max}")
print(f"Price y-limits: {y_min:.2f}% to {y_max:.2f}%")
print(f"Volume y-limit: 0 to {volume_max:.0f}")

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def create_filled_channel_image(day_data, cutoff_time, volume_max, price_min, price_max, output_size=(224, 224)):
    """
    Create an RGB image where each channel represents filled areas under the curves.
    Red channel: Volume (filled under curve)
    Green channel: Price before cutoff (filled under curve)
    Blue channel: Price after cutoff (filled under curve)
    """
    # Split data at cutoff
    before_cutoff = day_data[day_data['timestamp'] <= cutoff_time].copy()
    after_cutoff = day_data[day_data['timestamp'] >= cutoff_time].copy()
    
    # Initialize RGB image
    rgb_image = np.zeros((*output_size, 3))
    
    # Get time ranges
    min_time = day_data['timestamp'].min()
    max_time = day_data['timestamp'].max()
    total_minutes = (max_time - min_time).total_seconds() / 60
    
    # Calculate cutoff position
    cutoff_minutes = (cutoff_time - min_time).total_seconds() / 60
    cutoff_idx = int((cutoff_minutes / total_minutes) * output_size[1])
    
    def create_filled_area(x_values, y_values, width, height):
        """Helper function to create a filled area under the curve"""
        # Create a meshgrid for the image
        x = np.linspace(x_values[0], x_values[-1], width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Interpolate the curve
        curve_interp = np.interp(x, x_values, y_values)
        
        # Create the filled area
        filled = np.zeros((height, width))
        for i in range(width):
            # filled[:int(curve_interp[i] * height), i] = 1
            filled[height - int(curve_interp[i] * height):, i] = 1
            
        return filled
    
    # Process volume data (Red channel)
    volume_times = np.array([(t - min_time).total_seconds() / 60 for t in before_cutoff['timestamp']])
    volume_normalized = np.clip(before_cutoff['volume'].values / volume_max, 0, 1)
    rgb_image[:, :cutoff_idx, 0] = create_filled_area(
        volume_times,
        volume_normalized,
        cutoff_idx,
        output_size[0]
    )
    
    # Process price before cutoff (Green channel)
    price_before_times = volume_times
    price_before_normalized = np.clip((before_cutoff['price_change'].values - price_min) / (price_max - price_min), 0, 1)
    rgb_image[:, :cutoff_idx, 1] = create_filled_area(
        price_before_times,
        price_before_normalized,
        cutoff_idx,
        output_size[0]
    )
    
    # Process price after cutoff (Blue channel)
    price_after_times = np.array([(t - min_time).total_seconds() / 60 for t in after_cutoff['timestamp']])
    price_after_normalized = np.clip((after_cutoff['price_change'].values - price_min) / (price_max - price_min), 0, 1)
    rgb_image[:, cutoff_idx:, 2] = create_filled_area(
        price_after_times - cutoff_minutes,  # Adjust times to start from 0
        price_after_normalized,
        output_size[1] - cutoff_idx,
        output_size[0]
    )
    
    return rgb_image, cutoff_idx

def visualize_channels_separately(rgb_image, cutoff_idx):
    """
    Visualize each channel separately and the combined RGB image
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
    
    # Plot individual channels
    ax1.imshow(rgb_image[:, :, 0], cmap='Reds')
    ax1.set_title('Volume (Red Channel)')
    ax1.axis('off')
    
    ax2.imshow(rgb_image[:, :, 1], cmap='Greens')
    ax2.set_title('Price Before Cutoff (Green Channel)')
    ax2.axis('off')
    
    ax3.imshow(rgb_image[:, :, 2], cmap='Blues')
    ax3.set_title('Price After Cutoff (Blue Channel)')
    ax3.axis('off')
    
    # Plot combined RGB image
    ax4.imshow(rgb_image)
    ax4.set_title('Combined RGB Image')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.show()

def process_and_save_filled_stock_image(day_data, cutoff_time, volume_max, price_min, price_max, 
                                      output_size=(224, 224), output_path=None, show_visualization=True):
    """
    Process stock data and save as RGB image with filled areas under curves
    """
    # Calculate price changes if not already present
    if 'price_change' not in day_data.columns:
        first_price = day_data['close'].iloc[0]
        day_data['price_change'] = ((day_data['close'] - first_price) / first_price) * 100
    
    # Create RGB image
    rgb_image, cutoff_idx = create_filled_channel_image(
        day_data,
        cutoff_time,
        volume_max,
        price_min,
        price_max,
        output_size
    )
    
    # Generate output path if not provided
    if output_path is None:
        date_str = day_data['timestamp'].dt.date.iloc[0].strftime('%Y%m%d')
        output_path = f'filled_stock_rgb_{date_str}.png'
    
    # Save image
    plt.imsave(output_path, rgb_image)
    print(f"Saved image to: {output_path}")
    
    # Show visualization if requested
    if show_visualization:
        visualize_channels_separately(rgb_image, cutoff_idx)
    
    return rgb_image, cutoff_idx

# Test with one day
test_date = merged_df['timestamp'].dt.date.unique()[0]
day_data = merged_df[merged_df['timestamp'].dt.date == test_date].copy()

# Create and save the image with filled areas
cutoff_time = pd.Timestamp(f"{test_date} 12:30:00")
rgb_image, cutoff_idx = process_and_save_filled_stock_image(
    day_data,
    cutoff_time,
    volume_max,
    y_min,
    y_max,
    output_size=(224, 224),
    show_visualization=True  # This will show both individual channels and combined image
)

import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_stock_dataset(merged_df, volume_max, price_min, price_max, 
                        output_dir='stock_images', output_size=(224, 224)):
    """
    Create a dataset of stock images for all available dates
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique dates
    unique_dates = sorted(merged_df['timestamp'].dt.date.unique())
    
    print(f"Processing {len(unique_dates)} days of data...")
    
    # Process each day
    for date in tqdm(unique_dates):
        # Get data for this day
        day_data = merged_df[merged_df['timestamp'].dt.date == date].copy()
        
        # Skip if not enough data
        if len(day_data) < 10:  # Minimum data points threshold
            continue
            
        # Calculate price changes
        first_price = day_data['close'].iloc[0]
        day_data['price_change'] = ((day_data['close'] - first_price) / first_price) * 100
        
        # Create cutoff time
        cutoff_time = pd.Timestamp(f"{date} 12:30:00")
        
        try:
            # Generate RGB image
            rgb_image, _ = create_filled_channel_image(
                day_data,
                cutoff_time,
                volume_max,
                price_min,
                price_max,
                output_size
            )
            
            # Save image
            date_str = date.strftime('%Y%m%d')
            plt.imsave(os.path.join(output_dir, f'stock_{date_str}.png'), rgb_image)
            
        except Exception as e:
            print(f"Error processing {date}: {str(e)}")
            continue
    
    # Print summary
    n_images = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
    print(f"\nDataset creation complete!")
    print(f"Total images created: {n_images}")
    print(f"Dataset saved to: {output_dir}")
    
    return n_images

# Create the dataset
n_images = create_stock_dataset(
    merged_df,
    volume_max,
    y_min,
    y_max,
    output_dir='stock_images',
    output_size=(224, 224)
)