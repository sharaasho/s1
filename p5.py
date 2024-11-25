import pandas as pd
df=pd.read_csv('D:\\book1.csv')
print(df)
matched=df['cylinder']==8
new_df=df[matched]
print(new_df)
a=df.groupby(['model year']).count()
print(a)
