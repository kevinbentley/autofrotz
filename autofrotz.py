import subprocess
import anthropic
import time
import os
import select
from typing import Tuple, Optional, List
import requests
import json
import re
import ast
import llm_api
import qdrant
from sentence_transformers import SentenceTransformer
import random
import uuid
from flask import Flask, jsonify, Response, request, send_from_directory
from queue import Queue
import threading
import argparse
import openai
#from rdf import RDFStateManager
#import rdflib

app = Flask(__name__, static_folder='public', static_url_path='')

message_queue = Queue()

# Role-to-Voice Mapping. I'm not sure if these are specific to one account or if they're universal.
role_to_voice_id = {
    'user': 'bIHbv24MWmeRgasZH58o',
    'assistant': 'N2lVS1w4EtoT3dr4eOWO',
    # Add more roles and their corresponding voice IDs here
    # For example, we can make a different user role with a different voice
}

@app.route('/')
def index():
    return send_from_directory('public', 'index.html')

# Add explicit debug route for static files
@app.route('/<path:filename>')
def custom_static(filename):
    try:
        return send_from_directory(app.static_folder, filename)
    except Exception as e:
        app.logger.error(f'Error serving {filename}: {str(e)}')
        return f'Error: {str(e)}', 404


@app.route('/get_message', methods=['GET'])
def get_message():
    if not message_queue.empty():
        # Get the next message from the queue
        message = message_queue.get()
        return jsonify(message), 200
    else:
        # Return an error if the queue is empty
        return jsonify({'error': 'No messages left in the queue.'}), 404


@app.route('/get_text', methods=['GET'])
def get_text():
    try:
        message = message_queue.get()
        return jsonify(message), 200

    except Exception as error:
        print('Error:', error)
        return 'Server Error', 500

@app.route('/api/elevenlabs', methods=['POST'])
def elevenlabs():
    data = request.get_json()
    role = data.get('role')
    text = data.get('text')

    # Validate
    if not isinstance(text, str) or not text.strip():
        return jsonify({'error': 'Invalid or missing "text" parameter.'}), 400

    if not isinstance(role, str) or not role.strip():
        return jsonify({'error': 'Invalid or missing "role" parameter.'}), 400

    normalized_role = role.lower()
    # Select the voice ID based on the role
    voice_id = role_to_voice_id.get(normalized_role)

    if not voice_id:
        print(f'Role "{role}" not found. Using default voice.')
        return jsonify({'error': f'Unknown role: {role}'}), 400

    try:
        ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
        if not ELEVENLABS_API_KEY:
            return jsonify({'error': 'ELEVENLABS_API_KEY not set'}), 500

        headers = {
            'Content-Type': 'application/json',
            'xi-api-key': ELEVENLABS_API_KEY,
        }

        payload = {
            'text': text,
            'voice_settings': {
                'stability': 0.5,
                'similarity_boost': 0.75,
            },
        }

        response = requests.post(
            f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}',
            headers=headers,
            json=payload
        )

        if not response.ok:
            print(f'ElevenLabs API error: {response.text}')
            return jsonify({'error': response.text}), response.status_code

        # Return the audio data
        return Response(response.content, mimetype='audio/mpeg')
    except Exception as error:
        print('Error in ElevenLabs API call:', error)
        return jsonify({'error': 'Internal Server Error'}), 500




class SlackPoster:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        if(len(webhook_url)==0):
            self.enabled = False
        else:
            self.enabled = True
        
    def post_message(self, text: str, thread_ts: str = None) -> dict:
        if(not self.enabled):
            return "" 
        payload = {
            "text": text,
            "mrkdwn": True
        }
        # Threads aren't working for some reason
        if thread_ts:
            payload["thread_ts"] = thread_ts
            
        response = requests.post(
            self.webhook_url,
            data=json.dumps(payload),
            headers={'Content-Type': 'application/json'}
        )
        return response


def parse_map_string(map_string):
    # Wrap the string in brackets to make it a valid list of lists
    map_string = "[" + map_string + "]"
    
    # Convert the map string to a list of lists using ast.literal_eval
    map_list = ast.literal_eval(map_string)
    
    return map_list


def merge_maps(existing_map, new_map):
    for entry in new_map:
        if entry not in existing_map:
            existing_map.append(entry)
    return existing_map


def extract_items(item_string):
    # Use regex to find all the words inside the quotes
    match = re.search(r'Inventory\[(.*?)\]', item_string)
    if match:
        # Extract the items as a comma-separated string and split them into a list
        items = match.group(1).replace('"', '').split(', ')
        return items
    return []

# Function to add items to the inventory if they're not already there
def add_to_inventory(inventory, item_string):
    items = extract_items(item_string)
    for item in items:
        if item not in inventory:
            inventory.append(item)

def merge_graphs(main_graph, temp_graph):
    for s, p, o in temp_graph:
        main_graph.remove((s, p, None))
        main_graph.add((s, p, o))

class FrotzAIPlayer:
    def __init__(self, game_path: str, claude_api_key: str, slack_webhook_url: str):
        self.anthropic_client = anthropic.Anthropic(api_key=claude_api_key)
        #self.graph = rdflib.ConjunctiveGraph()
        self.game_path = game_path
        self.game_process = None
        self.game_history = []
        self.slack = SlackPoster(slack_webhook_url)
        self.thread_ts = None
        self.map = []
        self.items = []
        self.use_local_llm = False
        self.llm_endpoint = "http://172.16.4.30:9001"
        self.qcollection = "zork1-1"
        self.qclient = qdrant.get_client("172.16.4.30",6333)
        qdrant.make_collection(client=self.qclient,collection_name=self.qcollection,vecsize=1024)# vecsize is based on the sentencetransformer's size
        self.messages: List[dict] = []
        self.embedding_model = SentenceTransformer('thenlper/gte-large')
        self.current_tokens = 0
        self.max_tokens = 4096
        self.max_moves = 800
        self.llm="claude"#"local"#"openai"#"claude"
        # Initialize conversation context
        oldprompt1 = '''As you play the game, keep track of the map layout. Every time you discover a new room, print it out like this:
Map[[room1,n,room2],[room1,d,room3]]
Later, if are trying to decide where to go, use the action "get map" to see the known map..'''

        self.system_prompt = '''
You are playing an interactive text adventure game. Your goal is to explore, solve puzzles, and eventually win the game. Use standard text adventure commands like: look, inventory, examine X, take X, drop X, go north/south/east/west, etc.

Think outloud about your observations and strategy before taking an action. This will help you clarify your thoughts and make better decisions.
When you are ready to take an action, use this format:
Action["look"]

Important rules:
- Only perform one action per turn
- After a few failed attempts at solving a puzzle, you must leave the area and explore somewhere else, try a different puzzle.
'''
        oldprompt2 = '''
        
Also, to make this more fun, anytime the game describes a sound, let me know so I can try to recreate the audio:
Sound["The door creaks open"]

2. When encountering a puzzle, think through these steps:
   - What is the immediate obstacle?
   - What tools/items do you have that could help?
   - What information might you be missing?
   - Are there other areas you haven't fully explored?
   - How can I apply critical thinking and logic about the items and environment I've seen to solve the puzzle?

3. Maintain a structured memory of:
   - Unsolved puzzles you've encountered
   - Areas you haven't fully explored
   - Items that seem important but you don't know how to use yet
   - Approaches you've already tried that didn't work

Think about your strategy and print it out like this:
Thought["I should explore this room thoroughly before trying to solve the puzzle"]

When you solve a puzzle or overcome a challenge, record it like this:
Memory["The troll can be killed with the sword"]

When you're stuck, follow this priority list:
1. Review your memories and insights for clues you might have missed
2. Explore any previously noted unexplored areas
3. Re-examine items in your inventory for new uses
4. Move to a completely different area to find new puzzles or items

When you are ready to take an action, use this format:
Action["look"]

Important rules:
- Only perform one action per turn
- Don't save memories about temporary states like inventory
- After every 5 failed attempts at solving a puzzle, you must leave the area and explore somewhere else
- Keep track of the number of times you've tried similar solutions to avoid loops
'''
    '''As you play, store your knowledge about the world by printing it in RDF format. Use an underscore to replace spaces. For example:
RDF["ex:door_Opening
a ex:Knowledge ;
ex:hasButton ex:Black_Button .
"]
'''    
    def add_or_update_info(self, subject, predicate, object_):
        self.graph.add((subject, predicate, object_))


    def get_memory_embedding(self, memory):
        embeddings = self.embedding_model.encode([memory])
        return embeddings[0]
    
    def create_memory(self, memory) -> None:
        """Store a memory of something into the vector database."""
        e = self.get_memory_embedding(memory)
        #print(f"Creating memory: {memory}")
        qdrant.write_vector(self.qclient, self.qcollection, e, {"id":str(uuid.uuid4().hex) ,"text":memory})
        
    def suggest_memories(self, context) -> str:
        """Look up memories in the vector database, and return return them here."""
        # Don't look up memories if the context is too short
        if(len(context)<100):
            return None
        # We're only going to suggest a memory sometimes.
        dice = random.random()
        if(dice < 1):
            e = self.get_memory_embedding(context)
            #print(f"Searching for memory: {context}")
            results = qdrant.search(self.qclient, self.qcollection, e, 5)
            if(len(results)<1):
                return None
            # Now we'll randomly choose a memory to suggest.
            selection = random.randint(0,len(results)-1)
            if(results[selection].score < 0.95):
                return None
            return results[selection].payload["text"]
        return None    

    def start_game(self) -> None:
        """Launch the frotz process with the specified game."""
        self.game_process = subprocess.Popen(
            ['dfrotz', "-p", self.game_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        
        # Start new thread in Slack
        game_name = os.path.basename(self.game_path)
        response = self.slack.post_message(f"🎮 *Starting new game: {game_name}*\n_AI player powered by LLM")
        if 'ts' in response:
            self.thread_ts = response['ts']
        
        # Reset conversation context when starting new game
        self.messages = self.messages[:1]  # Keep only the system message
        
    def read_game_output(self, timeout: float = 2.0) -> str:
        """Read output from the game using select for timeout control."""
        output = []
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            reads, _, _ = select.select([self.game_process.stdout], [], [], 0.1)
            
            if not reads:
                if self.game_process.poll() is not None:
                    break
                continue
                
            try:
                chunk = os.read(self.game_process.stdout.fileno(), 4096)
                if not chunk:
                    break
                output.append(chunk.decode('utf-8', errors='ignore'))
            except (IOError, OSError):
                time.sleep(0.1)
                continue
                
            time.sleep(0.1)
            
        return ''.join(output)
    
    def send_command(self, command: str) -> str:
        """Send a command to the game and return the response."""

        match = re.search(r'Action\["(.*?)"\]', command)

        if match:
            command = match.group(1)
            #print(extracted_value)
        else:
            print("No action found")
            return "What is your next move?"        
        if(command == "get map"):
            return "Map so far " + self.get_map_str()
        if(command == "list items"):
            return "Known items " + str(self.items)
        if not command.endswith('\n'):
            command += '\n'
        try:
            os.write(self.game_process.stdin.fileno(), command.encode('utf-8'))
            time.sleep(0.2)
            return self.read_game_output()
        except IOError as e:
            error_msg = f"Error sending command: {e}"
            print(error_msg)
            self.slack.post_message(f"⚠️ {error_msg}", self.thread_ts)
            return ""
    
    def countContextTokens(self):
        total_history_tokens = 0
        for msg in self.messages:
            if(msg["role"] == "assistant"):
                total_history_tokens += llm_api.countTokens(msg["content"][0].text)     
            else:
                total_history_tokens += llm_api.countTokens(msg["content"])
        return total_history_tokens
    
    def trimContext(self):
        while(self.current_tokens > self.max_tokens):
            msg = self.messages.pop(1)
            if(msg["role"] == "assistant" and self.llm != "local"):
                self.current_tokens -= llm_api.countTokens(msg["content"][0].text)
            else:
                self.current_tokens -= llm_api.countTokens(msg["content"])


    def get_ai_action(self, game_state: str) -> str:
        """Get the next action from Claude based on the current game state and history."""
        
        # Add the game state to the conversation
        self.messages.append({
            "role": "user",
            "content": game_state#f"Current game state:\n{game_state}\n\nWhat is your next action?"
        })
        self.current_tokens += llm_api.countTokens(game_state)
        #trim the context
        self.trimContext()
        if(self.llm == "local"):
            resp = llm_api.getCompletion(self.llm_endpoint,self.messages, self.system_prompt, 1000, "llama-3.5B")
            self.messages.append({
                    "role": "assistant",
                    "content": resp
                })
            self.current_tokens += llm_api.countTokens(resp)
            return resp
        elif(self.llm == "claude"):
            
            try:
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=200,
                    temperature=0.7,
                    system=self.system_prompt,
                    messages=self.messages
                )
                
                # Add Claude's response to the conversation history
                self.messages.append({
                    "role": "assistant",
                    "content": response.content
                })
                self.current_tokens += llm_api.countTokens(response.content[0].text)
                return response.content[0].text
                
            except Exception as e:
                error_msg = f"Error getting AI response: {e}"
                print(error_msg)
                self.slack.post_message(f"⚠️ {error_msg}", self.thread_ts)
                # In case of error, try a simple "look" command as fallback
                return "look"
        elif(self.llm == "openai"):
            try:
                oai_messages = [{"role":"system", "content":self.system_prompt}]
                for msg in self.messages:
                    oai_messages.append(msg)
                response = openai.ChatCompletion.create(
                    model="gpt-4-turbo", #gpt-4o",
                    messages=oai_messages,
                    max_tokens=200,
                    n=1,
                    stop=None,
                    temperature=0.7,
                )
                self.messages.append({
                    "role": "assistant",
                    "content": response['choices'][0]['message']['content']
                })

                # Extract and print the assistant's reply
                return response['choices'][0]['message']['content']
                
            except Exception as e:
                error_msg = f"Error getting AI response: {e}"
                print(error_msg)
                self.slack.post_message(f"⚠️ {error_msg}", self.thread_ts)
                # In case of error, try a simple "look" command as fallback
                return "look"
    
    def format_for_slack(self, text: str) -> str:
        """Format text for Slack with proper escaping and formatting."""
        return f"```{text}```"
    
    def get_map_str(self):
        return str(self.map)
    
    def append_to_file(self,content: str, filename: str):
        with open(filename, 'a') as file:
            file.write(content + '\n')  # Appends content and adds a newline

    def play_game(self, max_turns: int = 800) -> None:
        """Play the game for a specified number of turns or until it ends."""
        print("Starting game...")
        self.start_game()
        
        print("Reading initial game state...")
        game_state = self.read_game_output()
        print(f"Initial state:\n{game_state}")
        self.game_history.append(("", game_state))
        self.slack.post_message(self.format_for_slack(game_state), self.thread_ts)
        
        for turn in range(max_turns):
            #print(f"\nGetting AI action for turn {turn + 1}...")
            action = self.get_ai_action(game_state)
            action = action.rstrip(">")
            action = action.strip('\n')
            action = action.rstrip(">")# remove trailing prompts
            print(f"{action}")
                        
            self.slack.post_message(f"🤖 *AI Action:* `{action}`", self.thread_ts)
            match = re.search(r'Map\["(.*?)"\]', action)
            map_str = ""
            if match:
                map_str = match.group(1)
            new_map = parse_map_string(map_str)
            # Merge with existing map, avoiding duplicates
            self.map = merge_maps(self.map, new_map)

            match = re.search(r'Item\["(.*?)"\]', action)
            if match:
                add_to_inventory(self.items, match.group(1))

            match = re.search(r'Memory\["(.*?)"\]', action)
            if match:
                self.create_memory(match.group(1))

            #rdfmatch = re.search(r'RDF\["([^"]*)"', action)
            
            #if(rdfmatch):
            #    print(f"RDF match found: {rdfmatch.group(1)}")
            #    tmpgraph = rdflib.Graph()
            #    tmpgraph.parse(data=rdfmatch.group(1), format="turtle")
            #    merge_graphs(self.graph,tmpgraph)
          
            #print("Sending command to game...")
            message_queue.put({"role": "user", "text": action})
            self.append_to_file(action, "game_transcript.txt")
            response = self.send_command(action)
            response = response.rstrip(">")
            response = response.strip('\n')
            self.append_to_file(response, "game_transcript.txt")  

            message_queue.put({"role": "assistant", "text": response})
            memory = self.suggest_memories(response)
            if memory is not None:
                response += f"\n\n🧠 *Memory:* {memory}\n"
            if( response is None):
                continue
            
            print(f"Response:\n{response}")
            
            self.slack.post_message(self.format_for_slack(response), self.thread_ts)
            
            game_state = response
            self.game_history.append((action, response))
            
            if "*** You have won ***" in response:
                win_msg = "🏆 Game won! The AI has achieved victory!"
                print(win_msg)
                self.slack.post_message(win_msg, self.thread_ts)
                break
            elif "You have died" in response:
                death_msg = "💀 Game over - restarting..."
                print(death_msg)
                self.slack.post_message(death_msg, self.thread_ts)
                self.game_process.terminate()
                self.game_history = []
                self.start_game()
                game_state = self.read_game_output()
                self.game_history.append(("", game_state))
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.game_process:
            try:
                self.game_process.terminate()
                self.game_process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.game_process.kill()
            
        self.slack.post_message("🔚 Game session ended", self.thread_ts)
            
    def save_transcript(self, filename: str) -> None:
        """Save the game history to a file."""
        with open(filename, 'w') as f:
            for action, response in self.game_history:
                if action:
                    f.write(f"> {action}\n")
                f.write(f"{response}\n")

def run_app(host, listen_port):
    app.run(debug=False, port=listen_port, host=host, use_reloader=False)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Autofrotz: AI-powered Z-machine player')
    parser.add_argument('-z', '--zfile', type=str, default="textplayer/games/zork1.z5", help='Specify the z-file')
    parser.add_argument('-v', '--verbose', default=True, action='store_true', help='Enable verbose mode')
    parser.add_argument('-s', '--slack',  default=False, action='store_true', help='Enable Slack integration')
    parser.add_argument('-p', '--port', type=int, default=3001, help='Server port (default: 3001)')
    parser.add_argument('-b', '--bind', type=str, default='0.0.0.0', help='Bind address (default: 0.0.0.0)')
    
    args = parser.parse_args()
    
    # Assigning parsed arguments to variables
    z_file = args.zfile
    verbose_mode = args.verbose
    slack_enabled = args.slack
    server_port = args.port
    bind_address = args.bind
    
    # Example usage of the variables
    if verbose_mode:
        print("Verbose mode is enabled.")
    print(f"Z-file: {z_file}")
    print(f"Slack Enabled: {slack_enabled}")
    print(f"Server Port: {server_port}")
    print(f"Bind Address: {bind_address}")


    flask_thread = threading.Thread(target=run_app, args=(bind_address, server_port))
    flask_thread.start()

    claude_api_key = os.getenv('ANTHROPIC_API_KEY')
    slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    openai.api_key = os.getenv('OPENAI_API_KEY')
    if not openai.api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    if not claude_api_key:
        raise ValueError("Please set the ANTHROPIC_API_KEY environment variable")
    if not slack_webhook_url:
        raise ValueError("Please set the SLACK_WEBHOOK_URL environment variable")
        
    game_path = z_file
    if not os.path.exists(game_path):
        raise ValueError(f"Game file not found: {game_path}")
    
    player = FrotzAIPlayer(game_path, claude_api_key, slack_webhook_url)
    try:
        player.play_game(max_turns=player.max_moves)
    finally:
        player.cleanup()
        player.save_transcript("finished_game_transcript.txt")