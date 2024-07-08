import pandas as pd

# Load the Excel file
file_path = '111.xlsx'
df = pd.read_excel(file_path)

# Rename the unnamed column to '姓名'
df.rename(columns={'Unnamed: 0': '姓名'}, inplace=True)

# Calculate the total score
total_score = df['成绩'].sum()

# Calculate the average score
average_score = df['成绩'].mean()

print("总成绩求和为：",total_score)
print("平均分：",average_score)
