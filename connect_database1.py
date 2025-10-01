import mysql.connector
mydb=mysql.connector.connect(
    host="localhost",
    user="root",
    passwd=""
)
print(mydb)

mycursor=mydb.cursor()
mycursor.execute("CREATE DATABASE students")
mycursor.execute("SHOW DATABASES")
for x in mycursor:
    print(x)

mycursor.execute("drop database test")

mydb=mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="students"
)
mycursor=mydb.cursor()
mycursor.execute("create table stu_table(id int auto_increment primary key' name varchar(50),age int)")
print(mydb)

mycursor.execute("SHOW TABLES")
for x in mycursor:
    print(x)