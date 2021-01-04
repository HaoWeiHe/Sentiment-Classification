
import pandas as pd
import csv

df = pd.read_csv("data.csv")
customer_num = df['ID'].count()

print( "Total customers: {}".format(customer_num))

print("Customers by city:")
sort_c = df.groupby('CITY').size()
for city, val in sort_c.items():
    print("{}: {}".format(city,val))

print("Customers by country:")
vc = df.groupby('COUNTRY').size()
for c, f in vc.items():
    print("{}: {}".format(c, f))
   
print("Country with the largest number of customers' contracts:")
country_contract = df.groupby('COUNTRY')["CONTRCNT"].sum().sort_values(ascending=False)

print("{} ({} contracts)".format(country_contract.keys()[0], country_contract[0]))
lst = set()
for row in df.itertuples(name='CITY'):
    if row.ID > 0: lst.add(row.CITY)
print("Unique cities with at least one customer:")
print(len(lst))

