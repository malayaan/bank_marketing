import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar

# Given RPP and Se values
RPP_values = [19/81, 26/81, 40/81, 81/81]
Se_values = [11/17, 15/17, 17/17, 17/17]

# Interpolate using spline
spline = UnivariateSpline(RPP_values, Se_values, k=2, s=0)
RPP_smooth = np.linspace(min(RPP_values), max(RPP_values), 1000)
Se_smooth = spline(RPP_smooth)

# Find RPP where Se = 0.80 using optimization
target_Se = 0.80

def objective_function(rpp):
    return (spline(rpp) - target_Se)**2

result = minimize_scalar(objective_function, bounds=(0, 1), method='bounded')
RPP_at_target_Se = result.x

# Plotting the LIFT curve
plt.figure(figsize=(10, 6))
plt.plot(RPP_smooth, Se_smooth, linestyle='-', color='blue')
plt.scatter(RPP_values, Se_values, color='blue', marker='o', zorder=5)  # Original points
plt.axvline(x=RPP_at_target_Se, color='orange', linestyle='--')  # Vertical line at RPP_at_target_Se
plt.xlabel('RPP (Rate of Positive Predictions)')
plt.ylabel('Se (Sensibility)')
plt.title('LIFT Curve: Se vs. RPP')
plt.grid(True)
plt.xlim([0, 1])
plt.ylim([0, 1.1])
plt.axline((0,0), slope=1, color='red', linestyle='--')  # Diagonal reference line
plt.legend(['LIFT Curve (Smooth)', 'Data Points', 'Se = 0.80', 'Random Selection'])
plt.show()

print(f"RPP when Se is approximately 0.80: {RPP_at_target_Se:.4f}")

# Given 10% RPP
target_RPP = 0.10

# Find the two nearest points to target_RPP
indices = np.searchsorted(RPP_values, target_RPP)

# Extract the two nearest RPP and Se values
RPP1, RPP2 = RPP_values[indices-1], RPP_values[indices]
Se1, Se2 = Se_values[indices-1], Se_values[indices]

# Interpolate Se at target_RPP
interpolated_Se = Se1 + (target_RPP - RPP1) * (Se2 - Se1) / (RPP2 - RPP1)

# Calculate LIFT
LIFT_10_percent = interpolated_Se / target_RPP
print("LIFT_10_percent=", LIFT_10_percent)

# Calculate normalized LIFT values: Se(s)/RPP(s)
normalized_LIFT_values = [Se/RPP for Se, RPP in zip(Se_values, RPP_values)]

# Plotting the normalized LIFT curve
plt.figure(figsize=(10, 6))
plt.plot(RPP_values, normalized_LIFT_values, marker='o', linestyle='-', color='green')
plt.xlabel('RPP (Rate of Positive Predictions)')
plt.ylabel('Normalized LIFT (Se/RPP)')
plt.title('Normalized LIFT Curve')
plt.grid(True)
plt.xlim([0, 1])
plt.ylim([0, max(normalized_LIFT_values) + 1])
plt.axhline(y=1, color='red', linestyle='--')  # Reference line for random selection
plt.legend(['Normalized LIFT Curve', 'Random Selection'])
plt.show()