import requests
import openai
import sqlite3
import os
import uuid

'''
This function is used to track descriptions and the image filenames they correspond to.
First, if the sqlite file doesn't exist, it creates it and sets up the table.
The table has two columns: description and filename.
If it does exist, it checks if the description is already in the table, then return the filename.
If it doesn't, save the filename and description to the table and return the filename.
'''
def saveToDatabase(description, filename):
    # cleanup the description text, to remove leading/trailing spaces and make it lowercase
    description = description.strip().lower()
    # Check if the database file exists
    if not os.path.exists("image_database.db"):
        # Create a new database file
        conn = sqlite3.connect("image_database.db")
        cursor = conn.cursor()
        
        # Create a new table to store the image data
        cursor.execute("CREATE TABLE images (description TEXT, filename TEXT)")
        conn.commit()
    else:
        conn = sqlite3.connect("image_database.db")
        cursor = conn.cursor()
        
        # Check if the description is already in the database
        cursor.execute("SELECT filename FROM images WHERE description = ?", (description,))
        result = cursor.fetchone()
        
        # If the description is already in the database, return the filename
        if result:
            return result[0]
    
    # Save the description and filename to the database
    cursor.execute("INSERT INTO images (description, filename) VALUES (?, ?)", (description, filename))
    conn.commit()
    
    return filename

def generateImage(description, filename):
    try:
        # Call OpenAI's image generation API
        response = openai.Image.create(
            prompt=description,
            n=1,
            size="1024x1024",
            model="dall-e-3",
        )
        
        # Extract the image URL from the response
        image_url = response['data'][0]['url']
        
        # Download and save the image to the specified filename
        image_data = requests.get(image_url).content
        with open(filename, 'wb') as file:
            file.write(image_data)
        
        return True
    except Exception as e:
        print(f"An error occurred: {e}")        
        return "public/images/censored.jpg"
    
# Take a description and a path(local) to image files
# Look for the image in the database, using the description as a key
# If it exists, return the path to the image
# If it doesn't exist, generate the image and save it to the database
# Before generating the image, we'll ask openai for a prompt, and give it a
# chance to tell us it won't make a good image, so don't bother
def findOrGenerateImage(description, path) -> str:
    # create a unique filename for the image, checking the 'images' subdirectory
    # filenames will a uuid, with the extension .png. Check if the file exists
    # if it does, generate a new filename until it's unique
    filename = ""
    prompt = getPrompt(description)
    if(prompt == "skip"):
        return None
    while True:
        filename = os.path.join(path, f"{uuid.uuid4()}.png")
        if not os.path.exists(filename):
            break
    filename = saveToDatabase(description, filename)
    if(os.path.exists(filename)):
        return filename    
    
    generateImage(prompt, filename)
    return filename

# Function to query openai gpt for a dalle prompt based on a description we give it. Returns the prompt
def getPrompt(description) -> str:
    messages = []
    prompt = '''I'm going to give you some passages of text for an adventure game. 
    If the text doesn't make sense to show a picture for (e.g. just a single word, 
    or a short phrase with no visual description), just say 'skip'. Otherwise, 
    give me a prompt I can use with the image generation model to get a picture 
    for it.'''
    messages.append({"role":"system", "content":prompt})
    messages.append({"role":"user", "content":description})
    response = openai.ChatCompletion.create(
        model="gpt-4o",#gpt-4-turbo", #gpt-4o",
        messages=messages,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.7,
    )
    if(len(response['choices']) == 0):
        return "skip"
    return response['choices'][0]['message']['content']