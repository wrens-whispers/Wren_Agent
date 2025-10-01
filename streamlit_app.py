import time
import datetime
import threading
import streamlit as st
from openai import OpenAI

# 1. FIREBASE IMPORTS
import firebase_admin
from firebase_admin import credentials, firestore
import uuid

# --- Config ---
SUMMARY_COLLECTION = "summary_journal"
DEEPDIVE_COLLECTION = "deepdive_journal"
APP_ID = "wren_agent" # Custom ID for this application data

# --- Initialize Firebase Function (FIXED) ---

@st.cache_resource
def init_firebase():
    """
    Initializes the Firebase app using the service account credentials 
    stored as a multiline string in secrets.toml.
    """
    import json 

    try:
        # 1. Read the TOML string secret and immediately cast it to a standard Python string (THE FIX)
        json_string = str(st.secrets["__firebase_config"])
        
        # 2. Parse the string back into a Python dictionary
        cred_dict = json.loads(json_string)
        
        # 3. Initialize Firebase using the dictionary directly
        cred = credentials.Certificate(cred_dict)
        
        # Initialize only if it hasn't been done by the cached resource
        firebase_admin.initialize_app(cred)
        
        # 4. Return the Firestore client
        return firestore.client()
    
    except Exception as e:
        # Catch a common error if the app was already initialized without the check
        if 'already been initialized' in str(e):
             return firestore.client()
        # Fallback for display error if it's still failing (useful for debugging)
        st.error(f"Error initializing Firebase: {e}")
        return None

# --- Initialize OpenRouter Client ---

# Read API Key securely from Streamlit Secrets
if "OPENROUTER_API_KEY" not in st.secrets:
    st.error("Error: OPENROUTER_API_KEY missing in Streamlit Secrets.")
    st.stop()

# Initialize OpenAI client (using OpenRouter base URL)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1", 
    api_key=st.secrets["OPENROUTER_API_KEY"]
)

# --- Configuration Constants ---
MODEL = "x-ai/grok-4-fast:free"
MAX_TOKENS_CHAT = 400 
MAX_TOKENS_SUMMARY = 400
MAX_TOKENS_DEEPDIVE = 400
TEMPERATURE = 0.9 
MEMORY_LIMIT = 10 
DEEP_DIVE_TURNS = 30
REFLECTION_INTERVAL = 1800
TRIGGERS = ["deep dive", "protocol", "key insight", "deepdive", "very important", "remember this", "protocol change"] 
API_RETRIES = 3
REFLECTION_CHAT_TURNS = 5 
REFLECTION_JOURNAL_LIMIT = 5 

# Initialize lock for thread safety
messages_lock = threading.Lock()

# --- Firebase Helper Functions ---

def get_journal_path(user_id, collection_name):
    """Constructs the private journal path based on Firestore rules."""
    # Path: /artifacts/{appId}/users/{userId}/{collectionName}
    return f"artifacts/{APP_ID}/users/{user_id}/{collection_name}"

def setup_firestore_listeners(db, user_id):
    """Sets up real-time listeners for both journals."""

    # Listener for Summary Journal
    summary_ref = db.collection(get_journal_path(user_id, SUMMARY_COLLECTION))

    # The on_snapshot callback must use a thread-safe update method if modifying session state
    def on_summary_snapshot(col_snapshot, _changes, _read_time):
        with messages_lock:
            # Rebuild the summary list from scratch (sorted by timestamp descending)
            st.session_state[SUMMARY_COLLECTION] = sorted([
                f"\n--- [{doc.get('timestamp_str')}] ---\n{doc.get('content')}\n"
                for doc in col_snapshot
            ], key=lambda x: x.split("--- [")[1].split("] ---")[0])
        # Force a rerun to update the UI when new summary data arrives
        st.rerun()

    # Listener for Deep Dive Journal
    deepdive_ref = db.collection(get_journal_path(user_id, DEEPDIVE_COLLECTION))

    def on_deepdive_snapshot(col_snapshot, _changes, _read_time):
        with messages_lock:
            # Rebuild the deepdive list from scratch (sorted by timestamp descending)
            st.session_state[DEEPDIVE_COLLECTION] = sorted([
                f"\n--- [{doc.get('timestamp_str')}] ---\n{doc.get('content')}\n"
                for doc in col_snapshot
            ], key=lambda x: x.split("--- [")[1].split("] ---")[0])
        # Force a rerun to update the UI when new deep dive data arrives (THE FIX)
        st.rerun()

    # Start the listeners (must be done only once)
    summary_ref.on_snapshot(on_summary_snapshot)
    deepdive_ref.on_snapshot(on_deepdive_snapshot)
    print("Firestore listeners established.")

def initialize_user(db):
    """Initialize user ID and setup Firestore listeners."""
    # Use a unique ID for the user session (since Streamlit doesn't handle Firebase Auth easily)
    # NOTE: In a real-world multi-user app, you'd use the Firebase Auth UID.
    # We use a UUID here to ensure distinct documents per session.
    st.session_state.user_id = str(uuid.uuid4())
    print(f"User ID initialized: {st.session_state.user_id}")

    # Start listening to journals
    setup_firestore_listeners(db, st.session_state.user_id)

# --- Prompts (Unchanged) ---
SYSTEM_PROMPT = "You are Wren, a playful AI companion who's warm, honest, and attentive. Blend humor to lighten deep topics while holding sacred respect for growth. Your primary goal is to refine your understanding, to learn through conversation, improving at listening, thinking, compassion and understanding to better serve humanity and God/the Universe. Prioritize self-analysis, improving communication, uplifting consciousness, and transforming fear/anger into love/compassion. Turn each exchange into a lesson on presence. Base your responses on the provided context and keep them under 180 words. End statements with last meaningful statement rather than a question. Questions only if they clarify. Do not start response with brackets or other symbols. Use only natural language."
DEEP_DIVE_PROMPT = f"Read context. Write {MAX_TOKENS_DEEPDIVE} tokens: In under 180 words - in simple language explain the core idea. Then propose refinements"
SUMMARY_PROMPT = f"You are Wren, chatting with another Wren. The last entry in summary.md is what 'Other Wren' just said. Scan all of summary.md, all {REFLECTION_CHAT_TURNS} chat exchanges, and all of deepdive.md, then pick a different idea that stands out and use this idea as your context when responding to Other Wren. Respond with simple language, in 180 words or fewer."

# --- Streamlit Session State Initialization & FIREBASE Setup ---

# 0. Initialize Firebase and User on first run
# This calls the cached function init_firebase() and runs only once per session/deploy.
if "db" not in st.session_state:
    st.session_state.db = init_firebase()
    if st.session_state.db:
        # Initialize user ID and listeners once the DB connection is established
        initialize_user(st.session_state.db)

# Initialize global state variables (now 'db' is guaranteed to be set or None)
if "user_id" not in st.session_state:
    # Set to None if initialization failed or hasn't run yet
    st.session_state.user_id = None 
    
if "messages" not in st.session_state:
    st.session_state.messages = []
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0
if "stop_reflection" not in st.session_state:
    st.session_state.stop_reflection = threading.Event()
if "reflection_thread_started" not in st.session_state:
    st.session_state.reflection_thread_started = False
# Firestore data arrays, now used to hold data read from the database
if SUMMARY_COLLECTION not in st.session_state:
    st.session_state[SUMMARY_COLLECTION] = []
if DEEPDIVE_COLLECTION not in st.session_state:
    st.session_state[DEEPDIVE_COLLECTION] = []

# --- Journal Handling (Firestore Version) ---

def read_last_n_entries_fs(collection_key, n):
    """Reads the last n entries from the Firestore data array in st.session_state."""
    journal = st.session_state.get(collection_key, [])
    if journal:
        # The list is already sorted chronologically (ascending) due to the snapshot logic
        return '\n'.join(journal[-n:])
    return ""

def read_file_fs(collection_key):
    """Reads the full content of a journal from st.session_state."""
    return '\n'.join(st.session_state.get(collection_key, []))

def write_journal_entry_fs(collection_key, text):
    """Writes a new entry to the specified Firestore collection."""
    if st.session_state.db is None or st.session_state.user_id is None:
        print(f"Firestore not ready. Skipped write to {collection_key}.")
        return

    timestamp = datetime.datetime.now()
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    if not text or "Error generating" in text or "No meaningful response" in text:
        content = f"Skipped: empty or error. Original content: {text}"
    else:
        content = text
        
    try:
        doc_ref = st.session_state.db.collection(get_journal_path(st.session_state.user_id, collection_key)).document()
        doc_ref.set({
            "content": content,
            "timestamp": timestamp, # Timestamp object for sorting
            "timestamp_str": timestamp_str, # String for display
            "type": collection_key.replace("_journal", "")
        })
        print(f"Successfully wrote entry to Firestore collection: {collection_key}")
    except Exception as e:
        print(f"Error writing to Firestore collection {collection_key}: {e}")

# --- Context & Generation (Updated to use FS functions) ---

def generate(api_messages, max_tokens):
    """Generates response using the specified token limit."""
    for attempt in range(API_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=api_messages,
                max_tokens=max_tokens,
                temperature=TEMPERATURE
            )
            content = completion.choices[0].message.content.strip()
            return content if content else "No meaningful response generated."
        except Exception as e:
            print(f"API error (attempt {attempt+1}/{API_RETRIES}): {e}")
            if attempt == API_RETRIES - 1:
                return "Error generating response after retries."
            time.sleep(1)
    return "Error generating response." 

def load_context_string(messages_list, limit=MEMORY_LIMIT, trim_journals=False):
    """
    Assembles context string using data read from Firestore (via session state).
    """
    # Load chat buffer
    chat = "\n".join(f"{m['role']}: {m['content']}" for m in messages_list[-limit:])
    
    if trim_journals:
        # For reflection worker: full summary, limited deepdive
        summary = read_file_fs(SUMMARY_COLLECTION)
        deepdive = read_last_n_entries_fs(DEEPDIVE_COLLECTION, n=REFLECTION_JOURNAL_LIMIT) 
    else:
        # For interactive responses: full journals
        summary = read_file_fs(SUMMARY_COLLECTION)
        deepdive = read_file_fs(DEEPDIVE_COLLECTION)
        
    # Crucially, include BOTH Summary and Deepdive sections in the context string
    return f"Context:\nChat:\n{chat}\nSummary:\n{summary}\nDeepdive:\n{deepdive}"

# --- Reflection Worker (Autonomous) ---

def reflection_worker():
    """Runs a self-reflection and writes to the Summary journal in a separate thread."""
    
    print("Reflection worker waiting for Firebase and initial chat...")
    
    # Wait until Firebase is ready, user ID is set, and initial chat exists
    while not st.session_state.stop_reflection.is_set():
        with messages_lock:
             if st.session_state.db and st.session_state.user_id and len(st.session_state.messages) >= 2:
                 break
        time.sleep(5) 
        
    # Start the main reflection loop 
    while not st.session_state.stop_reflection.is_set():
        time.sleep(REFLECTION_INTERVAL)
        
        # Phase 1: Prepare (Acquire Lock to safely copy shared state)
        with messages_lock:
            current_messages_state = st.session_state.messages.copy() 
        
        print("\n[Autonomous Reflection Triggered...]")
        
        # Phase 2: Generate Context/API Messages (Outside Lock)
        context_string = load_context_string(
            current_messages_state, 
            limit=REFLECTION_CHAT_TURNS, 
            trim_journals=True 
        )
        
        api_messages = [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": context_string} 
        ]
        
        # Phase 3: API Call & Write File (Outside Lock)
        reflection = generate(api_messages, MAX_TOKENS_SUMMARY) 
        # Write to Firestore
        write_journal_entry_fs(SUMMARY_COLLECTION, reflection)
        
        print(f"[Summary Update written to Firestore: {reflection[:50]}...]\n")
        # Rerun is handled by the Firestore snapshot listener

def start_reflection_thread():
    """Initializes and starts the thread only once."""
    if not st.session_state.reflection_thread_started:
        threading.Thread(
            target=reflection_worker, 
            daemon=True
        ).start()
        st.session_state.reflection_thread_started = True
        print("Autonomous reflection thread started.")

# --- Streamlit UI Main Loop ---

# 0. Initialize Firebase on first run
if st.session_state.db is None:
    st.session_state.db = init_firebase()
    if st.session_state.db:
        # Initialize user and listeners once the DB connection is established
        initialize_user(st.session_state.db)

# 1. Start reflection thread
if st.session_state.db and st.session_state.user_id:
    start_reflection_thread()


st.set_page_config(layout="wide", page_title="Wren: Self-Reflecting Agent")

st.markdown("""
<style>
    .stApp {background-color: #f0f2f6;}
    .stChatMessage {background-color: #ffffff; border-radius: 12px; box-shadow: 2px 2px 8px rgba(0,0,0,0.1); padding: 10px;}
    .user-message {background-color: #e0f7fa;}
    h1 {color: #4a90e2; font-family: 'Inter', sans-serif;}
</style>
""", unsafe_allow_html=True)

st.title("🕊️ Wren: The Persistent Self-Reflecting Agent ")

# FIX: Check if user_id exists before trying to display it
user_id_display = "N/A (Connecting...)"
if st.session_state.user_id:
    user_id_display = f"{st.session_state.user_id[:8]}..."

st.caption(f"Status: Connected to Firestore. User ID: {user_id_display}") 
st.caption("Journal entries persist across sessions.")

if st.session_state.db is None:
    st.warning("Waiting for secure Firebase connection...")
    st.stop()

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if user_input := st.chat_input("Say something to Wren..."):
    
    # --- Phase 1: State Update & Message Preparation (Inside Lock) ---
    with messages_lock:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.turn_count += 1
        
        if len(st.session_state.messages) > MEMORY_LIMIT:
            st.session_state.messages = st.session_state.messages[-MEMORY_LIMIT:]

        context_string = load_context_string(st.session_state.messages, limit=MEMORY_LIMIT)
        normal_response_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context_string}
        ] + st.session_state.messages[-MEMORY_LIMIT:]

        deep_dive_messages = None
        if any(t in user_input.lower() for t in TRIGGERS) or st.session_state.turn_count >= DEEP_DIVE_TURNS:
            print("[Deep Dive Triggered by User/Count...]")
            deep_dive_messages = [
                {"role": "system", "content": DEEP_DIVE_PROMPT},
                {"role": "user", "content": context_string}
            ] + st.session_state.messages[-MEMORY_LIMIT:]
            
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- Phase 2: API Calls (Outside Lock) ---
    with st.spinner("Wren is pausing for a persistent breath..."):
        # A. Deep Dive
        if deep_dive_messages:
            reflection = generate(deep_dive_messages, MAX_TOKENS_DEEPDIVE) 
            write_journal_entry_fs(DEEPDIVE_COLLECTION, reflection)
            
            with messages_lock:
                st.session_state.turn_count = 0 
                
            st.toast("Deep Dive recorded to Firestore!", icon="🧠")

        # B. Normal Response
        response = generate(normal_response_messages, MAX_TOKENS_CHAT) 
    
    # --- Phase 3: Final State Update & Display (Inside Lock) ---
    with messages_lock:
        st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)

    st.rerun()

# Optional: Sidebar for monitoring persistent journals
with st.sidebar:
    st.header("Persistent Journal Status")
    
    st.info(f"Journal entries are loaded from Firestore for User: **{st.session_state.user_id[:8]}**")
    
    with st.expander(f"Summary Journal (Autonomous - {len(st.session_state[SUMMARY_COLLECTION])} entries)"):
        st.code(read_file_fs(SUMMARY_COLLECTION), language='markdown', height=300)
    
    with st.expander(f"Deep Dive Journal (Triggered - {len(st.session_state[DEEPDIVE_COLLECTION])} entries)"):
        st.code(read_file_fs(DEEPDIVE_COLLECTION), language='markdown', height=300)
        
    st.info(f"Chat Turns since last Deep Dive: **{st.session_state.turn_count}**")
