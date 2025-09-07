import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("/home/tushara/Documents/Ashoka/Lab 4/Data/exp1/exp1A_data.csv", header = 0)

x_all = np.concatenate([data.iloc[:,0], data.iloc[:,2], data.iloc[:,4]])
y_all = np.concatenate([data.iloc[:,1], data.iloc[:,3], data.iloc[:,5]])

m, c = np.polyfit(x_all, y_all, 1)

plt.scatter(data.iloc[:,0:1], data.iloc[:,1:2], color = "red", label = "Trial 1")
plt.scatter(data.iloc[:,2:3], data.iloc[:,3:4], color = "green", label = "Trial 2")
plt.scatter(data.iloc[:,4:5], data.iloc[:,5:6], color = "blue", label = "Trial 3")

x_fit = np.linspace(min(x_all), max(x_all), 100)
y_fit = m*x_fit + c
plt.plot(x_fit, y_fit, "k-", label=f"Fit: y={m:.2f}x+{c:.2f}")

plt.xlabel("Current Supplied (A)")
plt.ylabel("Magnetic Filed Produced (Gauss)")
plt.legend()
plt.show()

x_val = 1.25
y_val = m*x_val + c
print(f"For x={x_val}, y={y_val:.2f}")

