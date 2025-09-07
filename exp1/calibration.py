import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data
data = pd.read_csv("/home/tushara/Documents/Ashoka/Lab 4/Data/exp1/exp1A_data.csv", header=0)

# Extract current (they're identical across trials)
I = data.iloc[:,0].values  

# Extract B values from three trials
B1, B2, B3 = data.iloc[:,1].values, data.iloc[:,3].values, data.iloc[:,5].values

# Compute mean and std across trials
B_mean = np.mean([B1, B2, B3], axis=0)
B_std = np.std([B1, B2, B3], axis=0, ddof=1)   # sample standard deviation

# Fit to mean data
m, c = np.polyfit(I, B_mean, 1)

# Scatter raw data
#plt.scatter(I, B1, color="red", alpha=0.6, label="Trial 1")
#plt.scatter(I, B2, color="green", alpha=0.6, label="Trial 2")
#plt.scatter(I, B3, color="blue", alpha=0.6, label="Trial 3")

# Plot mean with error bars
plt.errorbar(I, B_mean, yerr=B_std, fmt="o", markerfacecolor="red", markeredgecolor="red",ecolor="black",markersize = 3, capsize=4, label="Average with error bars")

# Plot fit line
x_fit = np.linspace(min(I), max(I), 200)
y_fit = m*x_fit + c
plt.plot(x_fit, y_fit, "k--", label=f"Fit: y={m:.2f}x+{c:.2f}")

plt.xlabel("Current Supplied (A)")
plt.ylabel("Magnetic Field Produced (Gauss)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

