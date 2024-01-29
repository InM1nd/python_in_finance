
1. Read the file 'ATX.csv' that you can find in the folder Course Material / data into a pandas DataFrame. Create a new column containing the absolute value of the daily relative price range defined as the difference between the High and Low prices divided by the previous day's Adjusted Close. Create another column containing the absolute values of the daily returns. Make a scatter plot of the price range against the absolute values of the returns.
2. Create an array A with 5 rows and 7 columns containing the numbers 1 to 35, where the first column contains the numbers 1 to 5, the second column the numbers 6 to 10 and so on. Create a second array B that results from swapping the first and the last column of the array A. Print the product of: 
      -  all of the elements in B
      - the elements in the third column of B
      - the elements in the last row of B

3. Create an array A with 4 rows and 6 columns containing numbers randomly drawn from a normal distribution with mean of 1 and a standard deviation of 2. Create a second array B containing a slice of A comprising the last two rows and the first four columns. Multiply B an A using matrix multiplication.
4. Write a function that takes a list of integers and returns the smallest even element of that list. If there is no even number in the list, the function should return None. If the list contains any non-integers, the function should print "Non-integer detected!" and continue with the next element of the list.
5. Read the file 'ATX.csv' that you can find in the folder Course Material / data into a pandas DataFrame, turning the Date column into a DatetimeIndex. Replace missing values with the last previous available value. Plot the time series of the Adjusted Close.
6. Read the files 'ATX.csv' and 'OMV.csv' that you can find in the folder Course Material / data into pandas DataFrames, turning the Date column into a DatetimeIndex. Estimate rolling market betas (where the ATX serves as the market index) for OMV by regressing OMV's daily returns on the daily returns of the ATX. Use a rolling window of 252 days. Plot the estimates of beta over time.
7. Write a function that takes a string, removes all commas and all words shorter than four characters from it, and returns the remaining string.
8. Read the file 'ATX.csv' that you can find in the folder Course Material / data into a pandas DataFrame, turning the Date column into a DatetimeIndex. Drop all columns of the DataFrame except for  the Adjusted Close. Change the frequency of the data from daily to monthly, using the values at the end of the month. Compute and print the mean and the standard deviation of the monthly returns.



Solution:
№1
``` python
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'C:/Users/zabol/Downloads/ATX.csv'

df = pd.read_csv(file_path)
df = df.dropna()

df['Relative_Price_Range'] = abs((df["High"] - df["Low"]) /df["Adj Close"].shift(1))
df["Daily_Return"] = df['Adj Close'].pct_change()
df['Absolute_Daily_Return'] = abs(df['Daily_Return'])

plt.scatter(df['Relative_Price_Range'], df['Absolute_Daily_Return'])

plt.xlabel('Relative Price Range')
plt.ylabel('Absolute Daily Returns')
plt.title('Scatter Plot: Price Range vs. Absolute Daily Returns')
plt.show()
```



№2
``` python
import numpy as np

A = np.arange(1, 36).reshape(5, 7, order='F')
B = A.copy()
B[:, [0, -1]] = B[:, [-1, 0]]

print(A)
print(B)

product_all_elements = np.prod(B, dtype=np.float64)
product_third_column = np.prod(B[:, 2])
product_last_row = np.prod(B[-1, :])

print("Product of all elements in B:", product_all_elements)
print("Product of the elements in the third column of B:", product_third_column) 
print("Product of the elements in the last row of B:", product_last_row)
```

**Tips:** If prod number is too big, need to use  dtype=np.float64 (This ensures that the product is calculated using a `float64` data type, which can handle a larger range of values.)

№3

``` python
mean_value = 1
std_deviation = 2

A = np.random.normal(loc=mean_value, scale=std_deviation, size=(4, 6))

print("Array A:", A)

B = A[-2:, :4]
print(B)
print("Array B:", B)

result = np.matmul(B, A)
print("Result of matrix multiplication (B @ A):", result)
```

Notes:  
- `loc` is the mean of the normal distribution.
- `scale` is the standard deviation of the normal distribution.
- `size` specifies the shape of the array.
Slicing: 
1. **Basic Slicing:**
    `# Select the first two rows and the first three columns 
    subset = array[:2, :3]`
    
2. **Using Negative Indices:**
    `# Select all rows except the last two and all columns except the last three 
    subset = array[:-2, :-3]`
    
3. **Slicing with a Step:**
    `# Select every second row and every second column 
    subset = array[::2, ::2]`
    
4. **Boolean Indexing:**
    `# Select elements that satisfy a condition (e.g., greater than 0) 
    subset = array[array > 0]`
    
5. **Using `numpy.s_` for Complex Slicing Operations:**
    `from numpy import s_  # Select a specific block of elements using 
    s_ subset = array[s_[1:3, 2:5]]`
    
6. **Indexing with Lists or Arrays of Indices:**
    `# Select specific rows using a list of indices indices = [0, 2, 3] 
    subset = array[indices, :]`


№4

``` python
def find_smallest_even(lst):
    smallest_even = None

    for element in lst:
        if not isinstance(element, int):
            print("Non-integer detected!")
            continue

        if element % 2 == 0:
            if smallest_even is None or element < smallest_even:
                smallest_even = element

    return smallest_even

# Example usage:
numbers = [3, 8, 'a', 5, 12, 10, 7, 4]
result = find_smallest_even(numbers)

if result is not None:
    print("The smallest even element is:", result)
else:
    print("No even number found in the list.")
```



№5

``` python
df_1 = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# Replace missing values with the last previous available value
df_1.fillna(method='ffill', inplace=True)

# Plot the time series of the Adjusted Close
plt.figure(figsize=(10, 6))
df_1['Adj Close'].plot(label='Adjusted Close', color='blue')
plt.title('Time Series of Adjusted Close')
plt.xlabel('Date')
plt.ylabel('Adjusted Close')
plt.legend()
plt.show()
```

№6
``` python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Read the CSV files into DataFrames
atx_df = pd.read_csv('Course Material/data/ATX.csv', parse_dates=['Date'], index_col='Date')
omv_df = pd.read_csv('Course Material/data/OMV.csv', parse_dates=['Date'], index_col='Date')

# Merge the two DataFrames on the Date index
merged_df = pd.merge(omv_df, atx_df[['Adj Close']], how='left', left_index=True, right_index=True, suffixes=('_OMV', '_ATX'))

# Calculate daily returns
merged_df['Return_OMV'] = merged_df['Adj Close_OMV'].pct_change()
merged_df['Return_ATX'] = merged_df['Adj Close_ATX'].pct_change()

# Estimate rolling market betas using a rolling window of 252 days
rolling_window = 252
rolling_betas = []

for i in range(rolling_window, len(merged_df)):
    window_data = merged_df.iloc[i - rolling_window:i]
    
    X = sm.add_constant(window_data['Return_ATX'])
    y = window_data['Return_OMV']

    model = sm.OLS(y, X).fit()
    rolling_betas.append(model.params['Return_ATX'])

# Create a DataFrame with the rolling betas and corresponding dates
rolling_betas_df = pd.DataFrame({'Date': merged_df.index[rolling_window:], 'Rolling_Beta': rolling_betas})
rolling_betas_df.set_index('Date', inplace=True)

# Plot the estimates of beta over time
plt.figure(figsize=(10, 6))
rolling_betas_df['Rolling_Beta'].plot(label='Rolling Beta', color='blue')
plt.title('Rolling Market Beta for OMV')
plt.xlabel('Date')
plt.ylabel('Rolling Beta')
plt.legend()
plt.show()
```


This code performs the following steps:

1. Reads the 'ATX.csv' and 'OMV.csv' files into pandas DataFrames.
2. Merges the two DataFrames on the 'Date' index.
3. Calculates daily returns for both OMV and ATX.
4. Estimates rolling market betas by performing rolling regressions using a window of 252 days.
5. Creates a DataFrame with the rolling betas and corresponding dates.
6. Plots the estimates of beta over time.

Please ensure that you have the required libraries installed (`pandas`, `numpy`, `statsmodels`, `matplotlib`). If not, you can install them using `pip install pandas numpy statsmodels matplotlib`.



№7 
``` python
def process_string(input_string):
    # Remove commas from the string
    string_without_commas = input_string.replace(',', '')

    # Split the string into words and filter out words shorter than four characters
    filtered_words = [word for word in string_without_commas.split() if len(word) >= 4]

    # Join the filtered words back into a string
    result_string = ' '.join(filtered_words)

    return result_string

# Example usage:
input_str = "Hello, world! This is a sample string with some short words."
output_str = process_string(input_str)

print("Input String:", input_str)
print("Output String:", output_str)
```


№8
``` python
import pandas as pd

# Read the CSV file into a DataFrame
file_path = 'Course Material/data/ATX.csv'
df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# Keep only the 'Adj Close' column
df = df[['Adj Close']]

# Resample data to monthly frequency, using end-of-month values
monthly_df = df.resample('M').last()

# Compute monthly returns
monthly_returns = monthly_df.pct_change()

# Compute and print the mean and standard deviation of monthly returns
mean_return = monthly_returns.mean().values[0]
std_dev_return = monthly_returns.std().values[0]

print("Mean of Monthly Returns:", mean_return)
print("Standard Deviation of Monthly Returns:", std_dev_return)
```

This code does the following:

1. Reads the 'ATX.csv' file into a pandas DataFrame, setting the 'Date' column as a datetime index.
2. Keeps only the 'Adj Close' column.
3. Resamples the data to monthly frequency using the last value of each month.
4. Computes monthly returns.
5. Calculates and prints the mean and standard deviation of the monthly returns.

Adjust the file path if the 'ATX.csv' file is in a different location.


NOTES 

1. File reading + plot graph

``` python
import pandas as pd
import matplotlib.pyplot as plt

# Чтение файла ATX.csv в DataFrame
df = pd.read_csv('Course Material/data/ATX.csv')

# Создание новых столбцов
df['PriceRange'] = abs((df['High'] - df['Low']) / df['Adj Close'].shift(1))
df['Returns'] = abs(df['Adj Close'].pct_change())

# Scatter plot
plt.scatter(df['PriceRange'], df['Returns'])
plt.xlabel('Price Range')
plt.ylabel('Returns')
plt.title('Scatter Plot of Price Range vs Returns')
plt.show()
```


 2.  Create and work with array
 
``` python
 import numpy as np
# Создание массива A
A = np.arange(1, 36).reshape(5, 7)

# Создание массива B
B = A.copy()
B[:, [0, -1]] = B[:, [-1, 0]]

# Печать результатов
print("Product of all elements in B:", np.prod(B))
print("Product of elements in the third column of B:", np.prod(B[:, 2]))
print("Product of elements in the last row of B:", np.prod(B[-1, :]))
```

3.  Matrix multiply
``` python
# Создание массива A
A = np.random.normal(loc=1, scale=2, size=(4, 6))

# Создание массива B
B = A[-2:, :4]
# Матричное умножение
result = np.dot(B, A)

# Печать результата
print("Result of matrix multiplication:\n", result)
```

4. Function for finding the minimum even element

``` python
def smallest_even(lst):
    min_even = None
    for num in lst:
        if isinstance(num, int):
            if num % 2 == 0:
                if min_even is None or num < min_even:
                    min_even = num
        else:
            print("Non-integer detected!")

    return min_even

# Пример использования:
numbers = [3, 5, 2, 8, 10, 'abc', 4, 6]
result = smallest_even(numbers)
print("Smallest even element:", result)
```


5. Time series processing

``` python
# Чтение файла ATX.csv в DataFrame
df = pd.read_csv('Course Material/data/ATX.csv', parse_dates=['Date'], index_col='Date')

# Замена пропущенных значений
df['Adj Close'].fillna(method='ffill', inplace=True)

# Построение временного ряда
plt.plot(df['Adj Close'])
plt.xlabel('Date')
plt.ylabel('Adjusted Close')
plt.title('Time Series of Adjusted Close')
plt.show()
```


6. Trailing bet estimates for OMV.

``` python
# Чтение файлов ATX.csv и OMV.csv в DataFrame
df_atx = pd.read_csv('Course Material/data/ATX.csv', parse_dates=['Date'], index_col='Date')
df_omv = pd.read_csv('Course Material/data/OMV.csv', parse_dates=['Date'], index_col='Date')

# Расчет ежедневных доходностей
df_atx['Daily Returns'] = df_atx['Adj Close'].pct_change()
df_omv['Daily Returns'] = df_omv['Adj Close'].pct_change()

# Оценка скользящих бет
window_size = 252
rolling_covariance = df_omv['Daily Returns'].rolling(window=window_size).cov(df_atx['Daily Returns'])
rolling_variance = df_atx['Daily Returns'].rolling(window=window_size).var()
rolling_beta = rolling_covariance / rolling_variance

# Построение графика беты с течением времени
plt.plot(rolling_beta, label='Rolling Beta')
plt.xlabel('Date')
plt.ylabel('Beta')
plt.title('Rolling Beta for OMV with respect to ATX')
plt.legend()
plt.show()
```

7. String processing function

``` python
def process_string(input_str):
    # Удаление запятых
    input_str = input_str.replace(',', '')
    
    # Удаление слов короче 4 символов
    words = input_str.split()
    filtered_words = [word for word in words if len(word) >= 4]
    
    # Возвращение оставшейся строки
    result_str = ' '.join(filtered_words)
    return result_str

# Пример использования:
input_string = "Hello, this is a sample string with some short words."
processed_string = process_string(input_string)
print("Processed String:", processed_string)
```


8. Convert data and calculate statistics for the month.
``` python

# Чтение файла ATX.csv в DataFrame
df = pd.read_csv('Course Material/data/ATX.csv', parse_dates=['Date'], index_col='Date')

# Оставление только столбца 'Adj Close'
df = df[['Adj Close']]

# Изменение частоты данных на месячную
df_monthly = df.resample('M').last()

# Вычисление месячных доходностей
monthly_returns = df_monthly.pct_change()

# Печать среднего и стандартного отклонения месячных доходностей
print("Mean of Monthly Returns:", monthly_returns.mean().values[0])
print("Standard Deviation of Monthly Returns:", monthly_returns.std().values[0])

```

9. Handling missing values in DataFrame:

``` python
# Замена пропущенных значений средним значением
df.fillna(df.mean(), inplace=True)

# Замена пропущенных значений медианой
df.fillna(df.median(), inplace=True)

# Удаление строк с пропущенными значениями
df.dropna(inplace=True)
```

10. Operations with dates in pandas:

``` python
# Выделение года, месяца и дня из индекса DataFrame
df['Year'] = df.index.year
df['Month'] = df.index.month
df['Day'] = df.index.day

# Фильтрация по дате
start_date = '2023-01-01'
end_date = '2023-12-31'
filtered_df = df.loc[start_date:end_date]
```

11. String processing in Python:

``` python 
# Разделение строки на список слов
sentence = "This is a sample sentence."
word_list = sentence.split()

# Объединение списка слов в строку
new_sentence = ' '.join(word_list)

# Поиск подстроки в строке
substring = 'sample'
if substring in sentence:
    print("Substring found!")

```

12. Grouping and aggregation in pandas:

``` python 
# Группировка по столбцу 'Year' и вычисление среднего значения для каждой группы
mean_by_year = df.groupby('Year')['Adj Close'].mean()

# Группировка по нескольким столбцам и вычисление среднего значения
mean_by_year_month = df.groupby(['Year', 'Month'])['Adj Close'].mean()
```


MORE NOTES

**1. Pandas DataFrame:**

```python
import pandas as pd
# Read a CSV file into a DataFrame
df = pd.read_csv('filename.csv')

# Create a new column in the DataFrame
df['new_column'] = ...

# Drop a column from the DataFrame
df.drop('column_name', axis=1, inplace=True)

# Filter rows in the DataFrame based on a condition
df = df[df['condition'] == True]

# Group the DataFrame by a specific column
df = df.groupby('column_name')

# Calculate statistics on a column in the DataFrame
mean_value = df['column_name'].mean()

```

  
**2. NumPy arrays:**

```python
import numpy as np

# Create a NumPy array
array = np.array([1, 2, 3, 4, 5])

# Reshape the array into a 2D array
array = array.reshape((2, 3))

# Transpose the array
array = array.T

# Calculate the mean of the array
mean_value = np.mean(array)
```

  

**3. Matplotlib for plotting:**

```python
import matplotlib.pyplot as plt

# Create a figure and axis
fig, ax = plt.subplots()

# Plot a line chart
ax.plot(x_values, y_values)

# Set the labels for the axes
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

# Show the plot
plt.show()
```


**4. Financial functions in NumPy and pandas:**

```python
import numpy as np
import pandas as pd

# Calculate the present value of a cash flow
present_value = np.pv(discount_rate, cash_flows)

# Calculate the internal rate of return of a project
irr = np.irr(cash_flows)

# Calculate the Sharpe ratio of a portfolio
sharpe_ratio = (portfolio_returns - risk_free_rate) / portfolio_volatility
```

**5. Data manipulation with pandas:**

```python
import pandas as pd

# Merge two DataFrames on a common column
merged_df = pd.merge(df1, df2, on='column_name')

# Join two DataFrames vertically
joined_df = pd.concat([df1, df2], ignore_index=True)

# Resample a DataFrame to a different frequency
resampled_df = df.resample('M').mean()

# Shift the values in a DataFrame by a certain number of periods
shifted_df = df.shift(periods=1)
```

**6. Time series analysis with statsmodels:**

```python
import statsmodels.api as sm

# Fit a linear regression model to data
model = sm.OLS(y, X).fit()

# Get the estimated coefficients and their standard errors
coefficients = model.params
standard_errors = model.bse

# Forecast future values using the model
forecast = model.forecast(steps=5)
```
