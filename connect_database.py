import mysql.connector

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd=""
)
print(mydb)

mycursor = mydb.cursor()
mycursor.execute("CREATE DATABASE Authorized_user")
mycursor.execute("SHOW DATABASES")
for x in mycursor:
    print(x)

mycursor.execute("drop database hello")

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="Authorized_user"
)
mycursor = mydb.cursor()
mycursor.execute("create table users(id int primary key, Name varchar(50), Age int, Address varchar(50))")
print(mydb)

mycursor.execute("SHOW TABLES")
for x in mycursor:
    print(x)

sql="INSERT INTO stu_table(id,name,age) values(%s,%s,%s)"
val = (1,"Ishwar",23)
mycursor.execute(sql,val)
mydb.commit()
print(mycursor.rowcount,"record inserted")
