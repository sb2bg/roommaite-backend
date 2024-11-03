import iris
import json

namespace = "USER"
port = "1972"
hostname = "localhost"
connection_string = f"{hostname}:{port}/{namespace}"
username = "demo"
password = "demo"

conn = iris.connect(connection_string, username, password)
cursor = conn.cursor()

# cursor.execute("DROP TABLE data.users")
# cursor.execute(
#     "CREATE TABLE data.users (uuid VARCHAR(255), location VARCHAR(255), answers VECTOR(DOUBLE, 384), prefs VECTOR(DOUBLE, 384))"
# )

# print current db schema
# cursor.execute("SELECT * FROM data.users")
# print(cursor.fetchall())


cursor.close()
