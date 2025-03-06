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
import afimages
#from rdf import RDFStateManager
#import rdflib

app = Flask(__name__, static_folder='public', static_url_path='')

message_queue = Queue()

# Role-to-Voice Mapping. I'm not sure if these are specific to one account or if they're universal.
role_to_voice_id = {
    'xplayer': 'bIHbv24MWmeRgasZH58o',
    'buddy': 'SAz9YHcvj6GT2YYXdXww',
    'player': 'bIHbv24MWmeRgasZH58o',
    'xbuddy': 'SPavHXefn4qr6bDvZI10',
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
                'stability': 0.8,
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
        self.buddy_mode = False
        self.reflection_mode = True
        self.map_mode = False
        self.game_path = game_path
        self.generate_images = False

        self.game_process = None
        self.game_history = []
        self.room_images = {}
        self.slack = SlackPoster(slack_webhook_url)
        self.thread_ts = None
        self.map = []
        self.items = []
        self.use_local_llm = False
        self.llm_endpoint = "http://172.16.4.30:9001"
        self.qcollection = "zork1-1"
        self.qclient = None #qdrant.get_client("172.16.4.30",6333)
        #qdrant.make_collection(client=self.qclient,collection_name=self.qcollection,vecsize=1024)# vecsize is based on the sentencetransformer's size
        self.messages: List[dict] = []
        self.embedding_model = SentenceTransformer('thenlper/gte-large')
        self.current_tokens = 0
        self.max_tokens = 4096
        self.max_moves = 800
        self.llm="openai"#"local"#"openai"#"claude"
        self.buddy_prompt = None
        self.summary_prompt = None
        self.summary = ""
        # Initialize conversation context

        with open('prompts/simple_prompt.txt', 'r') as file:
            self.system_prompt = file.read().strip()
        if(self.map_mode):
            with open('prompts/store_map_prompt.txt', 'r') as file:
                self.system_prompt += file.read().strip()
        if(self.reflection_mode):
            with open('prompts/reflection_prompt.txt', 'r') as file:
                self.system_prompt += file.read().strip()
        if(self.buddy_mode):
            with open('prompts/playing_with_buddy.txt', 'r') as file:
                self.system_prompt += file.read().strip()
            with open('prompts/buddy_prompt.txt', 'r') as file:
                self.buddy_prompt = file.read().strip()        
        with open('prompts/summary_prompt.txt', 'r') as file:
            self.summary_prompt = file.read().strip()

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
            match = re.search(r'Action:\s*\["(.*?)"\]', command)
            if match:
                command = match.group(1)
            else:
                print("No action found")
                return "Nothing happens."        
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
            #if(msg["role"] == "assistant" and self.llm != "local"):
            #    self.current_tokens -= llm_api.countTokens(msg["content"][0].text)
            #else:
            self.current_tokens -= llm_api.countTokens(msg["content"])

    def get_ai_completion(self, messages, system_prompt) -> str:
        if(self.llm == "local"):
            resp = llm_api.getCompletion(self.llm_endpoint,messages, system_prompt, 1000, "llama-3.5B")
            #self.current_tokens += llm_api.countTokens(resp)
            return resp
        elif(self.llm == "claude"):
            
            try:
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=200,
                    temperature=0.8,
                    system=system_prompt,
                    messages=messages
                )
                
                # Add Claude's response to the conversation history
                if(len(response.content)==0):
                    return ""
                return response.content[0].text
                
            except Exception as e:
                error_msg = f"Error getting AI response: {e}"
                print(error_msg)
                self.slack.post_message(f"⚠️ {error_msg}", self.thread_ts)
                # In case of error, try a simple "look" command as fallback
                return 'Action["look"]'
        elif(self.llm == "openai"):
            try:
                oai_messages = [{"role":"system", "content":system_prompt}]
                history_str = ""
                for msg in messages:
                    history_str += f'{msg["role"]}: {msg["content"]}\n'
                #    oai_messages.append(msg)
                oai_messages.append({"role":"user", "content":history_str})
                #for msg in messages:
                #    oai_messages.append(msg)
                response = openai.ChatCompletion.create(
                    model="gpt-4o",#gpt-4-turbo", #gpt-4o",
                    messages=oai_messages,
                    max_tokens=500,
                    n=1,
                    stop=None,
                    temperature=0.7,
                )

                # Extract and print the assistant's reply
                return response['choices'][0]['message']['content']
                
            except Exception as e:
                error_msg = f"Error getting AI response: {e}"
                print(error_msg)
                self.slack.post_message(f"⚠️ {error_msg}", self.thread_ts)
                # In case of error, try a simple "look" command as fallback
                return 'Action["look"]'
    def get_buddy_action(self) -> str:
        """Get the next action from the buddy based on the current game state and history. """
        

        return self.get_ai_completion(self.messages, self.buddy_prompt)
    
    def get_ai_action(self) -> str:
        """Get the next action from Claude based on the current game state and history."""
        
        return self.get_ai_completion(self.messages, self.system_prompt)
        
    def get_game_summary(self) -> str:
        """Use the llm to make a summary of the game so far."""
        return self.get_ai_completion(self.messages, self.summary_prompt)


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
        image_url = ""
        img = afimages.findOrGenerateImage(game_state, "public/images")
        if(img is not None):
            basename = os.path.basename(img)
            image_url = f"/images/{basename}"
        message_queue.put({"role": "assistant", "text": game_state, "image": image_url})#"/images/autofrotz.jpg"})
        print(f"Initial state:\n{game_state}")
        self.game_history.append(("", game_state))
        self.slack.post_message(self.format_for_slack(game_state), self.thread_ts)
        turn_number = 0
        for turn in range(max_turns):
            #print(f"\nGetting AI action for turn {turn + 1}...")
            # We won't trim the context here, so we might exceed the max contect this time.
            # Add the game state to the conversation
            self.messages.append({
                "role": "user",
                "content": game_state
            })
            self.append_to_file("Game: " + game_state, "game_transcript.txt")
            if(self.buddy_mode):
                suggestion = self.get_buddy_action()
                print(f"Buddy: {suggestion}")
                index = suggestion.find("[Player]")

                # Check if 'Player[' is in the string, sometimes the llm gets confused and the buddy tries to be the player too
                if index != -1:
                    # Slice the string up to 'Player[' (excluding 'Player[')
                    suggestion = suggestion[:index]
                suggestion = suggestion.rstrip(">")
                suggestion = suggestion.strip('\n')
                suggestion = suggestion.rstrip(">")
                suggestion = re.sub(r"^(?:\[Buddy\]:\s*)+", "", suggestion)

                self.messages.append({
                    "role": "assistant",
                    "content": "[Buddy]: " + suggestion
                })
                message_queue.put({"role": "Buddy", "text": suggestion})
                self.append_to_file("Buddy: " + suggestion, "game_transcript.txt")
                self.slack.post_message(f"🤦‍♀️ *Buddy comment:* `{suggestion}`", self.thread_ts)
                self.current_tokens += llm_api.countTokens(suggestion)
                self.trimContext()
            # Every 20 moves, we're going to ask for a summary of the game so far
            turn_number += 1
            if(turn_number % 30 == 0):
                self.summary += self.get_game_summary()
                print(f"Summary: {self.summary}")
                # Now we're going to clear the previous messages, and replace it with just the summary.
                self.messages = []
                self.messages.append({
                    "role": "user",
                    "content": self.summary
                })

            action = self.get_ai_action()
            if(len(action)==0):
                print("No action found")
                action = 'Action["look"]'
            #print(f"Action! [{action}]")
            action = action.rstrip(">")
            action = action.strip('\n')
            action = action.rstrip(">")# remove trailing prompts
            action = re.sub(r"^(?:\[Player\]:\s*)+", "", action)
            self.current_tokens += llm_api.countTokens(action)
            self.trimContext()
            self.messages.append({
                "role": "assistant",
                "content": "[Player]: " + action
            })
            match = re.search(r'Action\["(.*?)"\]', action)

            if not self.reflection_mode and match:
                message_queue.put({"role": "Player", "text": match.group(1)})
            else:
                message_queue.put({"role": "Player", "text": action})
            print(f"Player: {action}")
                        
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
          
            #message_queue.put({"role": "user", "text": action})
            self.append_to_file("Player: " + action, "game_transcript.txt")
            # send the command to the game
            response = self.send_command(action)
            response = response.replace(">", "")
            response = response.strip('\n')
            image_url = None
            if(self.generate_images):
                img = afimages.findOrGenerateImage(response, "public/images")
                if(img is not None):
                    basename = os.path.basename(img)
                    image_url = f"/images/{basename}"
                self.append_to_file("Game: " + response, "game_transcript.txt")  
                q = {"role": "assistant", "text": response}
                if(image_url is not None):
                    q["image"] = image_url
                message_queue.put(q)
            '''memory = self.suggest_memories(response)
            if memory is not None:
                response += f"\n\n🧠 *Memory:* {memory}\n"
            if( response is None):
                continue
            '''
            print(f"Game:\n{response}")
            
            self.slack.post_message(self.format_for_slack("🎮 " + response), self.thread_ts)
            
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