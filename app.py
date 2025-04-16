# Basic flask stuff for building http APIs and rendering html templates
from flask import Flask, render_template, redirect, url_for, request, session, jsonify

# Bootstrap integration with flask so we can make pretty pages
from flask_bootstrap import Bootstrap

# Flask forms integrations which save insane amounts of time
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField, TextAreaField, IntegerField, FloatField, SelectField, BooleanField
from wtforms.validators import DataRequired

# Basic python stuff
import os
import json
import functools
import datetime

# Some nice formatting for code
import misaka

# Need OpenAI for a few providers including local with llama.cpp server or ollama server
from openai import OpenAI

# Nice way to load environment variables for deployments
from dotenv import load_dotenv
load_dotenv()

# The total amount of message history in the chat
HISTORY_LENGTH = 8

# Create the Flask app object
app = Flask(__name__)

# Session key
app.config['SECRET_KEY'] = os.environ["SECRET_KEY"]
app.config['SESSION_COOKIE_NAME'] = 'bottybot2'

# BottyBot API Key for /api/chat endpoint
BOTTY_KEY = os.environ["BOTTY_KEY"]

# Configure all available models based on configured keys
models = []

# Local models are configured as a dict in .env, they need to start with "local-"
if "LOCAL_MODELS" in os.environ:
    local_models = json.loads(os.environ["LOCAL_MODELS"])
    for model_name in local_models:
        models.append(model_name)

# Configure MistralAI
if "MISTRAL_API_KEY" in os.environ: 
    models.append("mistral-small-latest")
    models.append("ministral-3b-latest")
    models.append("ministral-8b-latest")
    models.append("mistral-large-latest")
    from mistralai import Mistral
    mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# Configure OpenAI
if "OPENAI_API_KEY" in os.environ:
    models.append("gpt-4.1")
    models.append("gpt-4.1-mini")
    models.append("gpt-4.1-nano")
    models.append("o1-preview")
    models.append("o1-mini")
    oai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Configure Anthropic
if "ANTHROPIC_API_KEY" in os.environ:
    models.append("claude-3-5-haiku-latest")
    models.append("claude-3-5-sonnet-latest")
    models.append("claude-3-7-sonnet-latest")
    import anthropic
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Configure Cerebras
if "CEREBRAS_API_KEY" in os.environ:
    models.append("cerebras-llama3.1-8b")
    models.append("cerebras-llama-3.3-70b")
    from cerebras.cloud.sdk import Cerebras
    cerebras_client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

# Configure Google Gemini
if "GEMINI_API_KEY" in os.environ:
    models.append("gemini-2.0-flash-lite")
    models.append("gemini-2.0-flash")
    models.append("gemini-2.5-pro-exp-03-25")
    gemini_client = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url="https://generativelanguage.googleapis.com/v1beta/")

# Configure Cerebras
if "DEEPSEEK_API_KEY" in os.environ:
    models.append("deepseek-chat")
    models.append("deepseek-reasoner")
    deepseek_client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

# User Auth
users_string = os.environ["USERS"]
users = json.loads(users_string)

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
    if model_type.startswith("local-"):
        return llm_local(prompt, model_type, bot_config)
    if model_type.startswith("mistral-") or model_type.startswith("ministral-"):
        return llm_mistral(prompt, model_type, bot_config)
    if model_type.startswith("gpt-") or model_type.startswith("chatgpt-"):
        return llm_oai(prompt, model_type, bot_config)
    if model_type.startswith("o1"):
        return llm_o1(prompt, model_type, bot_config)
    if model_type.startswith("claude-"):
        return llm_anthropic(prompt, model_type, bot_config)
    if model_type.startswith("cerebras-"):
        return llm_cerebras(prompt, model_type, bot_config)
    if model_type.startswith("gemini-"):
        return llm_gemini(prompt, model_type, bot_config)
    if model_type.startswith("deepseek-"):
        return llm_deepseek(prompt, model_type, bot_config)

# Query local models
def llm_local(prompt, model_name, bot_config):
    client = OpenAI(api_key="doesntmatter", base_url=local_models[model_name])
    messages=[{"role": "system", "content": bot_config["identity"]},{"role": "user", "content": prompt}]
    response = client.chat.completions.create(max_tokens=4096, model=model_name, temperature=float(bot_config["temperature"]), messages=messages)
    user = bot_config["name"] + " " + model_name
    return {"user": user, "text": response.choices[0].message.content}

# Query mistral models
def llm_mistral(prompt, model_name, bot_config):
    messages=[{"role": "system", "content": bot_config["identity"]},{"role": "user", "content": prompt}]
    response = mistral_client.chat.complete(model=model_name, temperature=float(bot_config["temperature"]), messages=messages)
    user = bot_config["name"] + " " + model_name
    return {"user": user, "text": response.choices[0].message.content}

# Query OpenAI models
def llm_oai(prompt, model_name, bot_config):
    messages=[{"role": "system", "content": bot_config["identity"]},{"role": "user", "content": prompt}]
    response = oai_client.chat.completions.create(model=model_name, temperature=float(bot_config["temperature"]), messages=messages)
    user = bot_config["name"] + " " + model_name
    return {"user": user, "text": response.choices[0].message.content}

# Query OpenAI o1 models, without a system message.  O1 class models don't support identities or temperature.
def llm_o1(prompt, model_name, bot_config):
    messages=[{"role": "user", "content": prompt}]
    response = oai_client.chat.completions.create(model=model_name, messages=messages)
    user = bot_config["name"] + " " + model_name
    return {"user": user, "text": response.choices[0].message.content}

# Query Anthropic models
def llm_anthropic(prompt, model_name, bot_config):
    messages=[{"role": "user", "content": prompt}]
    response = anthropic_client.messages.create(system=bot_config["identity"], model=model_name, temperature=float(bot_config["temperature"]), messages=messages)
    user = bot_config["name"] + " " + model_name
    return {"user": user, "text": response.content[0].text}

# Query Cerebras models
def llm_cerebras(prompt, model_name, bot_config):
    model_name = model_name.replace("cerebras-", "")
    messages=[{"role": "system", "content": bot_config["identity"]},{"role": "user", "content": prompt}]
    response = cerebras_client.chat.completions.create(model=model_name, temperature=float(bot_config["temperature"]), messages=messages)
    user = bot_config["name"] + " " + model_name
    return {"user": user, "text": response.choices[0].message.content}

# Google Gemini
def llm_gemini(prompt, model_name, bot_config):
    messages=[{"role": "system", "content": bot_config["identity"]},{"role": "user", "content": prompt}]
    response = gemini_client.chat.completions.create(model=model_name, temperature=float(bot_config["temperature"]), messages=messages)
    user = bot_config["name"] + " " + model_name
    return {"user": user, "text": response.choices[0].message.content}

# Deepseek Chat (coding)
def llm_deepseek(prompt, model_name, bot_config):
    if model_name.endswith("-reasoner"):
        messages=[{"role": "user", "content": prompt}]
        response = deepseek_client.chat.completions.create(model=model_name, messages=messages)
    else:
        messages=[{"role": "system", "content": bot_config["identity"]},{"role": "user", "content": prompt}]
        response = deepseek_client.chat.completions.create(model=model_name, temperature=float(bot_config["temperature"]), messages=messages)
    user = bot_config["name"] + " " + model_name
    return {"user": user, "text": response.choices[0].message.content}

# Flask forms is magic
class PromptForm(FlaskForm):
    prompt = StringField('Prompt 💬', validators=[DataRequired()])
    model_type = SelectField('Model', choices=models, validators=[DataRequired()])
    raw_output = BooleanField('Raw Output')
    submit = SubmitField('Submit')

# Config form for bot
class BotConfigForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    identity = TextAreaField('Identity', validators=[DataRequired()])
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

    # Model will be the same as the last selected
    if "model_type" in session:
        form.model_type.data = session["model_type"]

    # Raw output option will be the same as the last selected
    if "raw_output" in session:
        form.raw_output.data = session["raw_output"]
    else:
        session["raw_output"] = False

    # Load the history array but remove items past 5
    history_file = session["user"] + "-history.json"
    history = load_chat_history(history_file)
    if len(history) > HISTORY_LENGTH:
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

        # Create the prompt with the chat history
        prompt = "Chat history:\n" + text_history(history) + "\n" + form_result["prompt"]

        # This new prompt is now history
        new_prompt = {"user": session["user"], "text": form_result["prompt"]}
        history.append(new_prompt)

        # Determine if we're using raw output
        if "raw_output" in form_result:
            session["raw_output"] = True
        else:
            session["raw_output"] = False

        # Prompt the LLM (with the augmentation), add that to history too!
        session["model_type"] = form_result["model_type"]
        new_history = llm_proxy(augmentation + prompt, bot_config, form_result["model_type"])
        if new_history == None:
            new_history = {"user": "error", "text": "Model Error 😭"}
        history.append(new_history)

        # Dump the history to the user file - multitenant!
        with open(history_file, 'w',  encoding='utf-8') as file:
            json.dump(history, file)
        return redirect(url_for('index'))
    
    # Spit out the template with either raw output or with each entry in the history formatted with Misaka (default)
    if session["raw_output"]:
        for dictionary in history:
            dictionary["text"] = dictionary["text"].replace('\n', '<br>')
        return render_template('index.html', history=history, form=form)
    else:
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
    form.temperature.data = bot_config["temperature"]

    if form.validate_on_submit():
        form_result = request.form.to_dict(flat=True)
        bot_config["name"] = form_result["name"]
        bot_config["identity"] = form_result["identity"]
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
@app.route('/api/chat', methods=['POST'])
def api_chat():

    # Validated API key
    api_key = request.form.get('api_key')  # Use request.form.get instead of request.args.get for POST requests
    if api_key != BOTTY_KEY:
        return jsonify({"error": "Invalid API Key"})

    # Get the API parameters and bail out if they're wrong
    prompt = request.form.get('prompt')
    model_type = request.form.get('model_type')
    if not prompt or not model_type:
        return jsonify({"error": "You need to send a prompt and the model name you want to use eg. llama-cpp"})

    # Yeah this is hacky but we want this to fail and load
    # the default wizard persona
    bot_config = load_bot_config("null")

    result = llm_proxy(prompt, bot_config, model_type)
    return jsonify(result)