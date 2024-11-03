from fastapi import FastAPI
import iris
from sentence_transformers import SentenceTransformer

# Load a pre-trained sentence transformer model. This model's output vectors are of size 384
model = SentenceTransformer("all-MiniLM-L6-v2")

namespace = "USER"
port = "1972"
hostname = "localhost"
connection_string = f"{hostname}:{port}/{namespace}"
username = "demo"
password = "demo"

app = FastAPI()


# "CREATE TABLE users (uuid VARCHAR(255), answers VECTOR(DOUBLE, 384), prefs VECTOR(DOUBLE, 384))"


@app.get("/find_matches/{prefs}/{n}/{location}")
def find_matches(prefs, n, location):
    sql = f"""
    SELECT TOP ? uuid
    FROM data.users
    WHERE location = ?
    ORDER BY VECTOR_DOT_PRODUCT(answers, TO_VECTOR(?)) DESC
    """

    conn = iris.connect(connection_string, username, password)
    cursor = conn.cursor()
    search_vector = model.encode(prefs, normalize_embeddings=True).tolist()
    cursor.execute(sql, [n, location, str(search_vector)])
    results = cursor.fetchall()
    return results


@app.get("/create_user/{uuid}/{answers}/{prefs}/{location}")
def create_user(uuid, answers, prefs, location):
    sql = f"""
    INSERT INTO data.users (uuid, location, answers, prefs)
    VALUES (?, ?, TO_VECTOR(?), TO_VECTOR(?))
    """

    answer_vector = model.encode(answers, normalize_embeddings=True).tolist()
    prefs_vector = model.encode(prefs, normalize_embeddings=True).tolist()

    conn = iris.connect(connection_string, username, password)
    cursor = conn.cursor()
    cursor.execute(sql, [uuid, location, str(answer_vector), str(prefs_vector)])
    result = conn.commit()
    conn.close()
    return {"response": result}


@app.get("/update_answers/{uuid}/{answers}")
def update_answers(uuid, answers):
    sql = f"""
    UPDATE data.users
    SET answers = TO_VECTOR(?)
    WHERE uuid = ?
    """

    answer_vector = model.encode(answers, normalize_embeddings=True).tolist()

    conn = iris.connect(connection_string, username, password)
    cursor = conn.cursor()
    cursor.execute(sql, [str(answer_vector), uuid])
    result = conn.commit()
    conn.close()
    return {"response": result}


@app.get("/update_prefs/{uuid}/{prefs}")
def update_prefs(uuid, prefs):
    sql = f"""
    UPDATE data.users
    SET prefs = TO_VECTOR(?)
    WHERE uuid = ?
    """

    prefs_vector = model.encode(prefs, normalize_embeddings=True).tolist()

    conn = iris.connect(connection_string, username, password)
    cursor = conn.cursor()
    cursor.execute(sql, [str(prefs_vector), uuid])
    result = conn.commit()
    conn.close()
    return {"response": result}


@app.get("/update_location/{uuid}/{location}")
def update_location(uuid, location):
    sql = f"""
    UPDATE data.users
    SET location = ?
    WHERE uuid = ?
    """

    conn = iris.connect(connection_string, username, password)
    cursor = conn.cursor()
    cursor.execute(sql, [location, uuid])
    result = conn.commit()
    conn.close()
    return {"response": result}
