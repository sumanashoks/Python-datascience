#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# Generate example data
x = np.linspace(0, 10, 100)  # Generate 100 data points between 0 and 10
y = 2 * x + 1 + np.random.randn(100)  # Linear relationship with some noise

# Plot the data points
plt.scatter(x, y, label='Data Points')

# Fit a linear regression line
coefficients = np.polyfit(x, y, 1)  # Fit a first-degree polynomial (linear regression)
trend_line = np.polyval(coefficients, x)  # Compute values for the trend line

# Plot the trend line
plt.plot(x, trend_line, color='red', label='Trend Line')

# Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Example Trend Analysis')
plt.legend()

# Show plot
plt.show()


# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data for different patterns
x = np.linspace(0, 10, 100)  # Generate 100 data points between 0 and 10

# Linear Pattern
y_linear = 2 * x + 1 + np.random.randn(100)  # Linear relationship with some noise

# Non-linear Pattern
y_non_linear = 0.5 * x**2 - 2*x + 3 + np.random.randn(100)  # Quadratic relationship with noise

# Periodic Pattern
y_periodic = np.sin(x) + np.random.randn(100)  # Sine wave with noise

# Seasonal Pattern
y_seasonal = np.sin(x) + np.sin(2*x) + np.random.randn(100)  # Sum of two sine waves with noise

# Cluster Pattern
x_cluster = np.concatenate((np.random.normal(0, 1, 50), np.random.normal(5, 1, 50)))
y_cluster = np.concatenate((np.random.normal(0, 1, 50), np.random.normal(5, 1, 50)))

# Outlier Pattern
y_outlier = np.concatenate((np.random.normal(0, 1, 90), np.array([10, 12, 15, 18, 20])))

# Plot the different patterns
plt.figure(figsize=(15, 10))

# Linear Pattern
plt.subplot(3, 3, 1)
plt.scatter(x, y_linear, label='Linear Pattern')
plt.title('Linear Pattern')

# Non-linear Pattern
plt.subplot(3, 3, 2)
plt.scatter(x, y_non_linear, label='Non-linear Pattern')
plt.title('Non-linear Pattern')

# Periodic Pattern
plt.subplot(3, 3, 3)
plt.scatter(x, y_periodic, label='Periodic Pattern')
plt.title('Periodic Pattern')

# Seasonal Pattern
plt.subplot(3, 3, 4)
plt.scatter(x, y_seasonal, label='Seasonal Pattern')
plt.title('Seasonal Pattern')

# Cluster Pattern
plt.subplot(3, 3, 5)
plt.scatter(x_cluster, y_cluster, label='Cluster Pattern')
plt.title('Cluster Pattern')

# Outlier Pattern
plt.subplot(3, 3, 6)
plt.scatter(range(len(y_outlier)), y_outlier, label='Outlier Pattern')
plt.title('Outlier Pattern')

plt.tight_layout()
plt.show()



# In[ ]:




