import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# !! 1(b) - Generate and visualise using synthetic dataset !!

#to generate synthetic performance data of monthly loan sales for 100 bank branches
np.random.seed(50)

sales = np.random.normal(loc=500000, scale=80000, size=100)

data = pd.DataFrame({
    "Branch": range(1,101),
    "Sales": sales
})

#summary statistics
print(data.head())
print(data["Sales"].describe()) #summary statistics of the sales column

mean_sales = data["Sales"].mean() #calculating arithmetic mean
median_sales = data["Sales"].median() #median
std_sales = data["Sales"].std() # calculating standard deviation for measuring spread of data
skew_sales = stats.skew(data["Sales"]) #calculating skewness to measure assymetry
kurt_sales = stats.kurtosis(data["Sales"], fisher=False) #calculating tailedness

print("Mean:", round(mean_sales, 2))
print("Median:", round(median_sales, 2))
print("Standard deviation:", round(std_sales, 2))
print("Skewness:", round(skew_sales, 2))
print("Kurtosis:", round(kurt_sales, 2))

#Histogram and density curve
plt.hist(data["Sales"], bins=12, density=True, edgecolor="black")

x = np.linspace(data["Sales"].min(), data["Sales"].max(), 100)
y = stats.norm.pdf(x, mean_sales, std_sales)

plt.plot(x, y)
plt.axvline(mean_sales, linestyle="--", label="Mean") # showing average value of monthly loan sales
plt.axvline(median_sales, linestyle=":", label="Median") #showing the middle sales value to compare with mean and assessing symmetry 

plt.title("Histogram and Normal Density Curve of Monthly Loan Sales")
plt.xlabel("Monthly Loan Sales")
plt.ylabel("Density")
plt.legend()
plt.show()

#Boxplot

plt.boxplot(data["Sales"].dropna(), vert=False, patch_artist=True)

plt.title("Box Plot of Monthly Loan Sales")
plt.xlabel("Monthly Loan Sales")
plt.show()

# !! 1(C) - Goodness of fit test !!

#Q-Q plot 
stats.probplot(data["Sales"].dropna(), dist="norm",plot=plt)
plt.title("Q-Q Plot of Monthly Loan Sales")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.show()

#Shapiro-Wilk Test
stat, p = stats.shapiro(data["Sales"].dropna())

print(f"Statistic={stat:.4f}, p-value={p:.4f}")

#Interpretation
if p > 0.05:
    print("The data appears approximately normally distributed (Fail to reject H0).")

else:
    print("The data is not normally distributed(Reject H0).")

