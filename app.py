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

# Some nice formatting for code
import misaka

# Nice way to load environment variables for deployments
from dotenv import load_dotenv
load_dotenv()

# Create the Flask app object
app = Flask(__name__)

# Session key
app.config['SECRET_KEY'] = os.environ["SECRET_KEY"]

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

# Output the whole history as a text blob
def text_history(history):
    text_history = ""
    for item in history:
        text_history = text_history + item["user"] + ": " + item["text"] + "\n"
    return text_history

def llm(user_prompt, bot_config):

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

    # HTML-ify the output
    #output = output.replace("\n", "<br>")
    output = misaka.html(output, extensions=misaka.EXT_FENCED_CODE)

    return {"user": bot_config["name"], "text": output}

# Flask forms is magic
class PromptForm(FlaskForm):
    prompt = StringField('Prompt 💬', validators=[DataRequired()])
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

    # Load the history array but remove items past 5
    history_file = session["user"] + "-history.json"
    history = load_chat_history(history_file)
    if len(history) > 5:
        history.pop(0)

    # Load the bot config
    bot_file = session["user"] + "-bot.json"
    bot_config = load_bot_config(bot_file)   

    # If user is prompting send it
    if form.validate_on_submit():
        # Get the form variables
        form_result = request.form.to_dict(flat=True)

        # Create history for the user's prompt
        new_history = {"user": session["user"], "text": form_result["prompt"]}
        history.append(new_history)
        prompt = text_history(history) + form_result["prompt"]

        # Prompt the LLM, add that to history too!
        new_history = llm(prompt, bot_config)
        history.append(new_history)

        # Dump the history to the user file - multitenant!
        with open(history_file, 'w',  encoding='utf-8') as file:
            json.dump(history, file)
        return redirect(url_for('index'))
    
    # Spit out the template
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