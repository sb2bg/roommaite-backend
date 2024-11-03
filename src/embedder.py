import iris
from sentence_transformers import SentenceTransformer

namespace = "USER"
port = "1972"
hostname = "localhost"
connection_string = f"{hostname}:{port}/{namespace}"
username = "demo"
password = "demo"

data = {
    "04948299-5d0c-4196-857b-8a9e4b6a78cb": "What is your age? 27; What is your education level? PhD; What is your occupation? Operational researcher; What is your major? Biology; What is your level of cleanliness? Not very clean; What is your level of noise tolerance? Low; What time do you usually go to bed? 10 PM; What time do you usually wake up? 7 AM; Do you smoke? No; Do you drink? No; Do you have any pets? None; Do you have any dietary restrictions? No; What is your preferred number of roommates? 4; What is your budget? $604 per month; What is your preferred move-in date? 2024-02-13; What is your preferred lease length? 1 year; What is your preferred neighborhood? Wyattfort",
    "1702721c-2aa1-47d3-9b18-1fa0e8e932be": "What is your age? 22; What is your education level? PhD; What is your occupation? Academic librarian; What is your major? Engineering; What is your level of cleanliness? Not very clean; What is your level of noise tolerance? High; What time do you usually go to bed? 10 PM; What time do you usually wake up? 7 AM; Do you smoke? No; Do you drink? No; Do you have any pets? Bird; Do you have any dietary restrictions? Yes; What is your preferred number of roommates? 2; What is your budget? $1117 per month; What is your preferred move-in date? 2024-07-31; What is your preferred lease length? 6 months; What is your preferred neighborhood? East Sarah",
    "2ac750f5-20c8-4689-bd3f-85d85c02a815": "What is your age? 29; What is your education level? PhD; What is your occupation? Naval architect; What is your major? Biology; What is your level of cleanliness? Not very clean; What is your level of noise tolerance? Moderate; What time do you usually go to bed? 9 PM; What time do you usually wake up? 6 AM; Do you smoke? Yes; Do you drink? No; Do you have any pets? Fish; Do you have any dietary restrictions? No; What is your preferred number of roommates? 4; What is your budget? $1376 per month; What is your preferred move-in date? 2024-04-22; What is your preferred lease length? Month-to-month; What is your preferred neighborhood? West Dustinmouth",
    "3aebd9b3-60f3-4430-b8c7-4cd826f2825f": "What is your age? 22; What is your education level? Undergraduate; What is your occupation? Structural engineer; What is your major? Business; What is your level of cleanliness? Not very clean; What is your level of noise tolerance? Low; What time do you usually go to bed? 9 PM; What time do you usually wake up? 9 AM; Do you smoke? Yes; Do you drink? Yes; Do you have any pets? Fish; Do you have any dietary restrictions? No; What is your preferred number of roommates? 4; What is your budget? $1371 per month; What is your preferred move-in date? 2024-01-25; What is your preferred lease length? 1 year; What is your preferred neighborhood? Bryanberg",
    "46c6cde8-3be3-47f6-98ef-e9cabf30cc62": "What is your age? 25; What is your education level? PhD; What is your occupation? Theatre stage manager; What is your major? Biology; What is your level of cleanliness? Not very clean; What is your level of noise tolerance? High; What time do you usually go to bed? 10 PM; What time do you usually wake up? 6 AM; Do you smoke? No; Do you drink? No; Do you have any pets? Bird; Do you have any dietary restrictions? No; What is your preferred number of roommates? 3; What is your budget? $1019 per month; What is your preferred move-in date? 2024-06-05; What is your preferred lease length? 1 year; What is your preferred neighborhood? Mitchellfort",
    "5a0666f5-c329-4af3-9ff0-b7a9dd2c12b0": "What is your age? 27; What is your education level? PhD; What is your occupation? Colour technologist; What is your major? Engineering; What is your level of cleanliness? Moderately clean; What is your level of noise tolerance? Low; What time do you usually go to bed? 9 PM; What time do you usually wake up? 6 AM; Do you smoke? Yes; Do you drink? Yes; Do you have any pets? None; Do you have any dietary restrictions? No; What is your preferred number of roommates? 4; What is your budget? $983 per month; What is your preferred move-in date? 2024-03-12; What is your preferred lease length? Month-to-month; What is your preferred neighborhood? Ellismouth",
    "5d68c93f-f8f5-4405-9e05-e7d187811a7d": "What is your age? 22; What is your education level? High School; What is your occupation? Psychotherapist, dance movement; What is your major? Business; What is your level of cleanliness? Very clean; What is your level of noise tolerance? High; What time do you usually go to bed? 11 PM; What time do you usually wake up? 7 AM; Do you smoke? No; Do you drink? Yes; Do you have any pets? Fish; Do you have any dietary restrictions? Yes; What is your preferred number of roommates? 1; What is your budget? $1167 per month; What is your preferred move-in date? 2024-07-12; What is your preferred lease length? 1 year; What is your preferred neighborhood? Johnton",
    "6216607f-5388-47a7-81bc-7137a43ac857": "What is your age? 29; What is your education level? PhD; What is your occupation? Technical brewer; What is your major? Biology; What is your level of cleanliness? Very clean; What is your level of noise tolerance? Low; What time do you usually go to bed? 9 PM; What time do you usually wake up? 9 AM; Do you smoke? No; Do you drink? No; Do you have any pets? None; Do you have any dietary restrictions? No; What is your preferred number of roommates? 1; What is your budget? $920 per month; What is your preferred move-in date? 2024-05-16; What is your preferred lease length? 1 year; What is your preferred neighborhood? North Donaldmouth",
    "64b92507-28ee-46b8-ac3b-9c61f22d1a1a": "What is your age? 20; What is your education level? Graduate; What is your occupation? Regulatory affairs officer; What is your major? Computer Science; What is your level of cleanliness? Moderately clean; What is your level of noise tolerance? Low; What time do you usually go to bed? 9 PM; What time do you usually wake up? 7 AM; Do you smoke? Yes; Do you drink? No; Do you have any pets? None; Do you have any dietary restrictions? Yes; What is your preferred number of roommates? 2; What is your budget? $696 per month; What is your preferred move-in date? 2024-08-01; What is your preferred lease length? 6 months; What is your preferred neighborhood? South Jennastad",
    "6e4d8dd9-030f-4b87-8dfd-a85e45cca7ed": "What is your age? 22; What is your education level? PhD; What is your occupation? Technical author; What is your major? Engineering; What is your level of cleanliness? Moderately clean; What is your level of noise tolerance? Low; What time do you usually go to bed? 10 PM; What time do you usually wake up? 6 AM; Do you smoke? Yes; Do you drink? Yes; Do you have any pets? None; Do you have any dietary restrictions? Yes; What is your preferred number of roommates? 2; What is your budget? $1117 per month; What is your preferred move-in date? 2024-07-31; What is your preferred lease length? 6 months; What is your preferred neighborhood? East Sarah",
    "70a38449-7a1c-4a36-89ad-d5653635ab56": "What is your age? 19; What is your education level? Undergraduate; What is your occupation? Barista; What is your major? Engineering; What is your level of cleanliness? Moderately clean; What is your level of noise tolerance? Low; What time do you usually go to bed? 9 PM; What time do you usually wake up? 6 AM; Do you smoke? Yes; Do you drink? Yes; Do you have any pets? Dog; Do you have any dietary restrictions? No; What is your preferred number of roommates? 1; What is your budget? $1157 per month; What is your preferred move-in date? 2024-03-18; What is your preferred lease length? 6 months; What is your preferred neighborhood? Ryanchester",
}


# Load a pre-trained sentence transformer model. This model's output vectors are of size 384
model = SentenceTransformer("all-MiniLM-L6-v2")

data = {x: model.encode(v, normalize_embeddings=True).tolist() for x, v in data.items()}


# convert data
data = [
    (
        k,
        "Austin",
        str(v),
        str(v),
    )
    for k, v in data.items()
]

sql = f"""
    INSERT INTO data.users (uuid, location, answers, prefs)
    VALUES (?, ?, TO_VECTOR(?), TO_VECTOR(?))
"""

conn = iris.connect(connection_string, username, password)
cursor = conn.cursor()
cursor.execute("DELETE FROM data.users")
print(cursor.executemany(sql, data))
print(conn.commit())
cursor.close()
