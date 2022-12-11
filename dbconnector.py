import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="qubit_user",
  password="qubit_2022",
  database = "schoolbooks"
 
)

mycursor = mydb.cursor()
qr =  "Introduction Python"
sql = "SELECT description,imageUrl  FROM libooks  WHERE MATCH (description) AGAINST ( '{}'  IN NATURAL LANGUAGE MODE) Limit 1".format(qr)

print(sql)
# mycursor.execute("SHOW DATABASES")
mycursor.execute(sql)
myresult = mycursor.fetchall()
a = myresult[0]

print(a)