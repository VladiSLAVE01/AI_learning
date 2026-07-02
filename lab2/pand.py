import pandas as pd

# Читаем CSV
df = pd.read_csv('titanic.csv')

# Предпросмотр данных
print("\nПредпросмотр данных")
print(df.head())

# Информация о данных
print("\nИнформация о данных")
print(df.info())

# Проверка на наличие NaN. Методы fillna и dropna возвращают новые Dataframe
print("\n Проверка на наличие NaN")
print(df.isna())

# Заполнение NaN
print("\nЗаполнение NaN")
df_fillnas = df.fillna(0)
print(df_fillnas.isna().sum()) # короткий и понятный вывод, что пробелы в записях отсутствуют

#Удаление строк с NaN
df_dropped = df.dropna()
print("\nNaN в каждом столбце:")
print(df_dropped.isna().sum())


## Part 3
# Выбор данных
# Выбор столбца по метке
print("\n 1. Выбор одного столбца по метке:")
age_column = df['Age']
print(age_column.head())

# Выбор нескольких столбцов
print("\n 2. Выбор нескольких столбцов")
subset = df[['Name','Age','Sex','Survived']]
print(subset.head())

# Выбор строк по индексу
print("\n 3. Выбор строк по индексу с помощью .loc[]:")
row_5 = df.loc[5]
print("Строка с индексом 5:")
print(row_5)

# Выбор по условию
print("\n 4. Выбор по условию")
men_over_30 = df[(df['Sex'] == 'male') & (df['Age'] > 30)]
print("Мужчины старше 30")
print(men_over_30.head())

## Сортировка данных
# Сортировка данных по значениям столбцов.
print("\n Сортировка данных по значениям столбцов.")
df_sorted_age = df.sort_values('Age')
print(df_sorted_age[['Name', 'Age', 'Fare']].head())

## Группировка данных
print("\n Доля выжавших")
survival = df.groupby('Pclass')['Survived'].mean()
print(survival)

## Часть 4
print("\n\n====== PART 4 ======")
# 1. Читаем данные
df = pd.read_csv('titanic.csv')

# 2. Проверяем пропуски
print(df.isna().sum())

# 3. Заполняем нулями
df = df.fillna(0)

# 4. Первые 10 строк
print(df.head(10))

# 5. Пассажиры старше 30
age_over_30 = df[df['Age'] > 30]
print(age_over_30[['Name', 'Age', 'Pclass', 'Fare']].head(10))

# 6. Сортировка по Fare
df_sorted = df.sort_values('Fare', ascending=False)
print(df_sorted[['Name', 'Fare', 'Pclass']].head(10))

# 7. Средний возраст по классам
print(df.groupby('Pclass')['Age'].mean())