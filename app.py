# Basic flask stuff for building http APIs and rendering html templates
from flask import Flask, render_template, redirect, url_for, request, session, send_from_directory

# Bootstrap integration with flask so we can make pretty pages
from flask_bootstrap import Bootstrap

# Flask forms integrations which save insane amounts of time
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField, TextAreaField, IntegerField, FloatField, SelectField
from wtforms.validators import DataRequired

# Basic python stuff
import os
import json
import functools
import requests
import time
import datetime

# Import OpenAI and Mistral libraries
from openai import OpenAI
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import anthropic

# Some nice formatting for code
import misaka

# Nice way to load environment variables for deployments
from dotenv import load_dotenv
load_dotenv()

# Create the Flask app object
app = Flask(__name__)

# Session key
app.config['SECRET_KEY'] = os.environ["SECRET_KEY"]

# BottyBot API Key for /api/chat endpoint
BOTTY_KEY = os.environ["BOTTY_KEY"]

# Start with just a local model
models = [
    "llama-cpp"
]

# optionally connect the clients
if "MISTRAL_API_KEY" in os.environ:    
    models.append("mistral-tiny")
    models.append("mistral-small")
    models.append("mistral-medium")
    models.append("mistral-large")
    mistral_client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])

if "OPENAI_API_KEY" in os.environ:
    models.append("gpt-3.5-turbo")
    models.append("gpt-4-turbo")
    models.append("gpt-4o")
    models.append("gpt-4")
    oai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

if "ANTHROPIC_API_KEY" in os.environ:
    models.append("claude-3-opus-20240229")
    models.append("claude-3-sonnet-20240229")
    models.append("claude-3-haiku-20240307")
    anthropic_client =anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# User Auth
users_string = os.environ["USERS"]
users = json.loads(users_string)

# Load the llm model config
with open("model.json", 'r',  encoding='utf-8') as file:
    model = json.load(file)

# Make it pretty because I can't :(
Bootstrap(app)

# Load the chat history array
# Chat history looks like an array of events like {"user": "blah", "text": "How do I thing?"}
def load_chat_history(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, return an empty history
        data = []
    except json.JSONDecodeError:
        # Handle JSON decoding errors if the file contains invalid JSON
        print(f"Error decoding JSON in file: {file_path}")
        data = []
    return data

# Load the current bot config
def load_bot_config(file_path):

    # Our default Wizard persona.  Use this if there's no user defined config.
    data = {
        "name": "Wizard 🧙", 
        "identity": "You are Wizard, a friendly chatbot. You help the user answer questions, solve problems and make plans.  You think deeply about the question and provide a detailed, accurate response.", 
        "tokens": "-1", 
        "temperature": "0.7"
    }

    # Load our user configured bot config if there is one.
    try:
        with open(file_path, 'r',  encoding='utf-8') as file:
            data = json.load(file)
    except:
        pass

    return data

# Load the bot library - A library of useful bots to talk to about different subjects
def load_bot_library():
    data = []
    with open("bots.json", 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Load the augmentation file if there is one. This is used to augment the prompt with additional data
def load_augmentation(file_path):
    data = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except:
        pass
    return data

# Output the whole history as a text blob
def text_history(history):
    text_history = ""
    for item in history:
        text_history = text_history + item["user"] + ": " + item["text"] + "\n"
    return text_history

def llm_proxy(prompt, bot_config, model_type):
    if model_type == "llama-cpp":
        return llm(prompt, model_type, bot_config)
    if model_type.startswith("mistral-"):
        return llm_mistral(prompt, model_type, bot_config)
    if model_type.startswith("gpt-"):
        return llm_oai(prompt, model_type, bot_config)
    if model_type.startswith("claude-"):
        return llm_anthropic(prompt, model_type, bot_config)

# Query mistral models
def llm_mistral(prompt, model_name, bot_config):
    messages = [ChatMessage(role="system", content=bot_config["identity"]), ChatMessage(role="user", content=prompt)]
    response = mistral_client.chat(model=model_name, temperature=float(bot_config["temperature"]), messages=messages)
    user = bot_config["name"] + " " + model_name
    return {"user": user, "text": response.choices[0].message.content}

# Query OpenAI models
def llm_oai(prompt, model_name, bot_config):
    messages = [ChatMessage(role="system", content=bot_config["identity"]), ChatMessage(role="user", content=prompt)]
    response = oai_client.chat.completions.create(model=model_name, temperature=float(bot_config["temperature"]), messages=messages)
    user = bot_config["name"] + " " + model_name
    return {"user": user, "text": response.choices[0].message.content}

# Query Anthropic models
def llm_anthropic(prompt, model_name, bot_config):

    message = anthropic_client.messages.create(
        model=model_name,
        max_tokens=4096,
        temperature=float(bot_config["temperature"]),
        system=bot_config["identity"],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )  
    user = bot_config["name"] + " " + model_name
    return {"user": user, "text": message.content[0].text}

def llm(user_prompt, model_name, bot_config):

     # Build the prompt
    prompt = model["prompt_format"].replace("{system}", bot_config["identity"])
    prompt = prompt.replace("{prompt}", user_prompt)

    api_data = {
        "prompt": prompt,
        "n_predict": int(bot_config["tokens"]),
        "temperature": float(bot_config["temperature"]),
        "stop": model["stop_tokens"],
        "tokens_cached": 0
    }

    # Attempt to do a completion but retry and back off if the model is not ready
    retries = 3
    backoff_factor = 1
    while retries > 0:
        try:
            response = requests.post(model["llama_endpoint"], headers={"Content-Type": "application/json"}, json=api_data)
            json_output = response.json()
            output = json_output['content']
            break
        except:
            time.sleep(backoff_factor)
            backoff_factor *= 2
            retries -= 1
            output = "My AI model is not responding try again in a moment 🔥🐳"
            continue

    # Why do you put your own name in the damn output?!
    output = output.lstrip(bot_config["name"] + ":\n")

    user = bot_config["name"] + " " + model_name
    return {"user": user, "text": output}

# Flask forms is magic
class PromptForm(FlaskForm):
    prompt = StringField('Prompt 💬', validators=[DataRequired()])
    model_type = SelectField('Model', choices=models, validators=[DataRequired()])
    submit = SubmitField('Submit')

# Config form for bot
class BotConfigForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    identity = TextAreaField('Identity', validators=[DataRequired()])
    tokens = IntegerField('Output Token Limit', validators=[DataRequired()])
    temperature = FloatField('LLM Temperature', validators=[DataRequired()])
    submit = SubmitField('Save')

# Bot library drop down and selection form
class BotLibraryForm(FlaskForm): 
    bot = SelectField('Select Premade Bot', choices=[], validators=[DataRequired()]) 
    load_bot = SubmitField('Load')

# Augmentation edit/clear form
class AugmentationForm(FlaskForm): 
    augmentation = TextAreaField('Augmentation')
    save = SubmitField('Save')
    clear = SubmitField('Clear')

# Amazing, I hate writing this stuff
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

# Define a decorator to check if the user is authenticated
# No idea how this works...  Magic.
def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if users != None:
            if session.get("user") is None:
                return redirect(url_for('login'))
        return view(**kwargs)        
    return wrapped_view

# The default chat view
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():

    # The single input box and submit button
    form = PromptForm()

    if "model_type" in session:
        form.model_type.data = session["model_type"]

    # Load the history array but remove items past 5
    history_file = session["user"] + "-history.json"
    history = load_chat_history(history_file)
    if len(history) > 5:
        history.pop(0)

    # Load the bot config
    bot_file = session["user"] + "-bot.json"
    bot_config = load_bot_config(bot_file)

    # Load the augmentation
    augment_file = session["user"] + "-augment.json"
    augmentation = load_augmentation(augment_file)  

    # If user is prompting send it
    if form.validate_on_submit():
        # Get the form variables
        form_result = request.form.to_dict(flat=True)

        # Create history for the user's prompt
        new_history = {"user": session["user"], "text": form_result["prompt"]}
        history.append(new_history)
        prompt = text_history(history) + form_result["prompt"]

        # Prompt the LLM (with the augmentation), add that to history too!
        session["model_type"] = form_result["model_type"]
        new_history = llm_proxy(augmentation + prompt, bot_config, form_result["model_type"])
        # Use Misaka library to format the output
        history.append(new_history)

        # Dump the history to the user file - multitenant!
        with open(history_file, 'w',  encoding='utf-8') as file:
            json.dump(history, file)
        return redirect(url_for('index'))
    
    # Spit out the template with formatted strings
    for dictionary in history:
        dictionary["text"] = misaka.html(dictionary["text"], extensions=misaka.EXT_FENCED_CODE)
    return render_template('index.html', history=history, form=form)

# Configure the bot
@app.route('/config', methods=['GET', 'POST'])
@login_required
def config():

    bot_file = session["user"] + "-bot.json"
    bot_config = load_bot_config(bot_file)    
    form = BotConfigForm()
    
    # Populate the form
    form.name.data = bot_config["name"]
    form.identity.data = bot_config["identity"]
    form.tokens.data = bot_config["tokens"]
    form.temperature.data = bot_config["temperature"]

    if form.validate_on_submit():
        form_result = request.form.to_dict(flat=True)
        bot_config["name"] = form_result["name"]
        bot_config["identity"] = form_result["identity"]
        bot_config["tokens"] = form_result["tokens"]
        bot_config["temperature"] = form_result["temperature"]
        with open(bot_file, 'w',  encoding='utf-8') as file:
            json.dump(bot_config, file)
        return redirect(url_for('index'))

    return render_template('config.html', form=form)

# Configure the prompt augmentation
@app.route('/augment', methods=['GET', 'POST'])
@login_required
def augment():
    augment_file = session["user"] + "-augment.json"
    augmentation = load_augmentation(augment_file)  
    form = AugmentationForm()

    # Populate the form
    form.augmentation.data = augmentation

    # Save the augmentation on a per user basis
    if form.validate_on_submit():
        form_result = request.form.to_dict(flat=True)
        # Clear the file or store it
        if "clear" in form_result:
            try:
                os.remove(augment_file)
            except:
                pass
        else:
            with open(augment_file, 'w',  encoding='utf-8') as file:
                json.dump(form_result["augmentation"], file)
        return redirect(url_for('index'))

    return render_template('augment.html', form=form)

# Bot Library
@app.route('/library', methods=['GET', 'POST'])
@login_required
def library():
    form = BotLibraryForm()

    # Populate the bot library drop down
    bot_library = load_bot_library()
    for bot in bot_library:
        form.bot.choices.append(bot["name"])

    # What config do we write to?
    bot_file = session["user"] + "-bot.json"

    if form.validate_on_submit():
        form_result = request.form.to_dict(flat=True)
        bot_selected = form_result["bot"]
        for dict_item in bot_library:
            if dict_item["name"] == bot_selected:
                bot_config = dict_item
                break
        with open(bot_file, 'w',  encoding='utf-8') as file:
            json.dump(bot_config, file)
        return redirect(url_for('config'))

    return render_template('library.html', form=form)
    
# Delete chat history, new chat
@app.route('/new')
@login_required
def new():
    history_file = session["user"] + "-history.json"
    try:
        os.remove(history_file)
    except:
        pass
    return redirect(url_for('index'))

# Delete bot identity, return to Wizard
@app.route('/reset')
@login_required
def reset():
    history_file = session["user"] + "-bot.json"
    try:
        os.remove(history_file)
    except:
        pass
    return redirect(url_for('index'))

# Download the chat history
@app.route('/backup')
@login_required
def backup():

    # We could have multiple bots in history but this is fine.
    bot_file = session["user"] + "-bot.json"
    bot_config = load_bot_config(bot_file)

    # Load the history to output for export
    history_file = session["user"] + "-history.json"
    history = load_chat_history(history_file)

    # Get the current date to tag to the export
    current_date = datetime.datetime.now()
    formatted_date = current_date.strftime('%Y-%m-%d')

    # Spit out the template with formatted strings
    for dictionary in history:
        dictionary["text"] = misaka.html(dictionary["text"], extensions=misaka.EXT_FENCED_CODE)
    return render_template('history.html', history=history, user=session["user"], bot=bot_config["name"], date=formatted_date)

# Login/logout routes that rely on the user being stored in session
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if form.username.data in users:
            if form.password.data == users[form.username.data]:
                session["user"] = form.username.data
                return redirect(url_for('index'))
    return render_template('login.html', form=form)

# We finally have a link for this now!
@app.route('/logout')
def logout():
    session["user"] = None
    return redirect(url_for('login'))

# Basic Chat API for some scripts to consume, right now only supports wizard persona
@app.route('/api/chat')
def api_chat():

    # Validated
    api_key = request.args.get('api_key')
    if api_key != BOTTY_KEY:
        return {"error": "Invalid API Key"}

    # Get the API parameters and bail out if they're wrong
    prompt = request.args.get('prompt')
    model_type = request.args.get('model_type')
    if not prompt or not model_type:
        return {"error": "You need to send a prompt and the model name you want to use eg. llama-cpp"}
    
    # Yeah this is hacky but we want this to fail and load
    # the default wizard persona
    bot_config = load_bot_config("null")
    result = llm_proxy(prompt, bot_config, model_type)
    return result