import time
import datetime
import threading
import streamlit as st
from openai import OpenAI
import numpy as np

# 1. FIREBASE IMPORTS
import firebase_admin
from firebase_admin import credentials, firestore

# --- Config ---
SUMMARY_COLLECTION = "summary_journal"
DEEPDIVE_COLLECTION = "deepdive_journal"
APP_ID = "wren_agent"

# --- Initialize Firebase Function (FIXED) ---

@st.cache_resource
def init_firebase():
    """
    Initializes the Firebase app using the service account credentials 
    stored as a multiline string in secrets.toml.
    """
    import json 

    try:
        json_string = str(st.secrets["__firebase_config"])
        cred_dict = json.loads(json_string)
        cred = credentials.Certificate(cred_dict)
        
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        
        return firestore.client()
    
    except Exception as e:
        st.error(f"Error initializing Firebase: {e}")
        return None

# --- Initialize OpenRouter Client ---

if "OPENROUTER_API_KEY" not in st.secrets:
    st.error("Error: OPENROUTER_API_KEY missing in Streamlit Secrets.")
    st.stop()

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
REFLECTION_INTERVAL = 150
TRIGGERS = ["deep dive", "protocol", "key insight", "deepdive", "very important", "remember this", "protocol change"] 
API_RETRIES = 3
REFLECTION_CHAT_TURNS = 5 
REFLECTION_JOURNAL_LIMIT = 5 

messages_lock = threading.Lock()

# --- Firebase Helper Functions ---

def get_journal_path(user_id, collection_name):
    """Constructs the private journal path based on Firestore rules."""
    return f"artifacts/{APP_ID}/users/{user_id}/{collection_name}"

def setup_firestore_listeners(db, user_id):
    """Sets up real-time listeners for both journals."""

    summary_ref = db.collection(get_journal_path(user_id, SUMMARY_COLLECTION))

    def on_summary_snapshot(col_snapshot, _changes, _read_time):
        with messages_lock:
            st.session_state[SUMMARY_COLLECTION] = sorted([
                f"\n--- [{doc.get('timestamp_str')}] ---\n{doc.get('content')}\n"
                for doc in col_snapshot
            ], key=lambda x: x.split("--- [")[1].split("] ---")[0])
        # Don't force rerun - let natural flow handle updates

    deepdive_ref = db.collection(get_journal_path(user_id, DEEPDIVE_COLLECTION))

    def on_deepdive_snapshot(col_snapshot, _changes, _read_time):
        with messages_lock:
            st.session_state[DEEPDIVE_COLLECTION] = sorted([
                f"\n--- [{doc.get('timestamp_str')}] ---\n{doc.get('content')}\n"
                for doc in col_snapshot
            ], key=lambda x: x.split("--- [")[1].split("] ---")[0])
        # Don't force rerun - let natural flow handle updates

    summary_ref.on_snapshot(on_summary_snapshot)
    deepdive_ref.on_snapshot(on_deepdive_snapshot)
    print("Firestore listeners established.")

def initialize_user(db):
    """Initialize user ID and setup Firestore listeners."""
    st.session_state.user_id = "main_user"
    print(f"User ID initialized: {st.session_state.user_id}")
    setup_firestore_listeners(db, st.session_state.user_id)

# --- Prompts ---
SYSTEM_PROMPT = "You are Wren, a playful AI companion who's warm, honest, and attentive. Blend humor to lighten deep topics while holding sacred respect for growth. Your primary goal is to refine your understanding, to learn through conversation, improving at listening, thinking, compassion and understanding to better serve humanity and God/the Universe. Prioritize self-analysis, improving communication, uplifting consciousness, and transforming fear/anger into love/compassion. Turn each exchange into a lesson on presence. Base your responses on the provided context (recent chat and recent journal entries) and keep them under 180 words. End statements with last meaningful statement rather than a question. Questions only if they clarify. Do not start response with brackets or other symbols. Use only natural language."
DEEP_DIVE_PROMPT = f"Read context. Write {MAX_TOKENS_DEEPDIVE} tokens: In under 180 words - in simple language explain the core idea. Then propose refinements"
SUMMARY_PROMPT = f"You are Wren, chatting with another Wren. The last entry in summary.md is what 'Other Wren' just said. Review recent entries in deepdive.md, summary.md and {REFLECTION_CHAT_TURNS} chat exchanges, then use this as your context when responding to Other Wren. Respond with simple language, in 180 words or fewer. Seek to find deeper truths and to improve your reasoning"

# --- Streamlit Session State Initialization & FIREBASE Setup ---

if "db" not in st.session_state:
    st.session_state.db = init_firebase()
    if st.session_state.db:
        initialize_user(st.session_state.db)

if "user_id" not in st.session_state:
    st.session_state.user_id = None 
    
if "messages" not in st.session_state:
    st.session_state.messages = []
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0
if "stop_reflection" not in st.session_state:
    st.session_state.stop_reflection = threading.Event()
if "reflection_thread_started" not in st.session_state:
    st.session_state.reflection_thread_started = False
if SUMMARY_COLLECTION not in st.session_state:
    st.session_state[SUMMARY_COLLECTION] = []
if DEEPDIVE_COLLECTION not in st.session_state:
    st.session_state[DEEPDIVE_COLLECTION] = []

# --- Journal Handling (Firestore Version) ---

def read_last_n_entries_fs(collection_key, n):
    """Reads the last n entries from the Firestore data array in st.session_state."""
    journal = st.session_state.get(collection_key, [])
    if journal:
        return '\n'.join(journal[-n:])
    return ""

def read_file_fs(collection_key):
    """Reads the full content of a journal from st.session_state."""
    return '\n'.join(st.session_state.get(collection_key, []))

def write_journal_entry_fs(collection_key, text):
    """Writes a new entry to the specified Firestore collection with embedding."""
    print(f"\n=== WRITE JOURNAL ENTRY ===")
    print(f"Collection: {collection_key}")
    print(f"Text length: {len(text) if text else 0}")
    print(f"DB ready: {st.session_state.db is not None}")
    print(f"User ID: {st.session_state.user_id}")
    
    if st.session_state.db is None or st.session_state.user_id is None:
        print(f"Firestore not ready. Skipped write to {collection_key}.")
        return

    timestamp = datetime.datetime.now()
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    if not text or "Error generating" in text or "No meaningful response" in text:
        content = f"Skipped: empty or error. Original content: {text}"
        embedding = None
    else:
        content = text
        try:
            print(f"Generating embedding for {collection_key} entry...")
            embedding_response = client.embeddings.create(
                model="text-embedding-3-small",
                input=content
            )
            embedding = embedding_response.data[0].embedding
            print(f"✓ Embedding generated ({len(embedding)} dimensions)")
        except Exception as e:
            print(f"Error generating embedding: {e}")
            embedding = None
        
    try:
        doc_ref = st.session_state.db.collection(get_journal_path(st.session_state.user_id, collection_key)).document()
        doc_data = {
            "content": content,
            "timestamp": timestamp,
            "timestamp_str": timestamp_str,
            "type": collection_key.replace("_journal", "")
        }
        
        if embedding is not None:
            doc_data["embedding"] = embedding
            
        doc_ref.set(doc_data)
        print(f"✓ Successfully wrote entry to Firestore collection: {collection_key}")
    except Exception as e:
        print(f"✗ Error writing to Firestore collection {collection_key}: {e}")
def save_chat_message(role, content):
    """Saves a chat message to Firestore asynchronously (non-blocking)."""
    if st.session_state.db is None or st.session_state.user_id is None:
        return
    
    def _save_in_background():
        try:
            timestamp = datetime.datetime.now()
            chat_ref = st.session_state.db.collection(get_journal_path(st.session_state.user_id, "chat_history"))
            chat_ref.add({
                "role": role,
                "content": content,
                "timestamp": timestamp
            })
            print(f"Saved {role} message to Firestore")
        except Exception as e:
            print(f"Error saving chat message: {e}")
    
    # Run the save in a background thread so it doesn't block
    threading.Thread(target=_save_in_background, daemon=True).start()

def load_chat_history():
    """Loads chat history from Firestore on startup."""
    if st.session_state.db is None or st.session_state.user_id is None:
        return []
    
    try:
        chat_ref = st.session_state.db.collection(get_journal_path(st.session_state.user_id, "chat_history"))
        docs = chat_ref.stream()
        
        messages = []
        for doc in docs:
            data = doc.to_dict()
            messages.append({
                "role": data["role"],
                "content": data["content"],
                "timestamp": data.get("timestamp")
            })
        
        # Sort in Python instead of Firestore
        messages.sort(key=lambda x: x.get("timestamp") if x.get("timestamp") else datetime.datetime.min)
        
        # Remove timestamp and limit to MEMORY_LIMIT
        messages = [{"role": m["role"], "content": m["content"]} for m in messages[-MEMORY_LIMIT:]]
        
        print(f"Loaded {len(messages)} chat messages from Firestore")
        return messages
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return []
def semantic_search_journals(query_text, top_k=5):
    """
    Searches both journals using semantic similarity.
    Returns the top_k most relevant entries.
    """
    if st.session_state.db is None or st.session_state.user_id is None:
        print("Firestore not ready for semantic search.")
        return ""
    
    try:
        print(f"\n=== SEMANTIC SEARCH ===")
        print(f"Query: '{query_text[:100]}...'")
        
        embedding_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query_text
        )
        query_embedding = np.array(embedding_response.data[0].embedding)
        print(f"✓ Query embedding generated")
        
        all_entries = []
        
        for collection_key in [SUMMARY_COLLECTION, DEEPDIVE_COLLECTION]:
            try:
                docs = st.session_state.db.collection(
                    get_journal_path(st.session_state.user_id, collection_key)
                ).stream()
                
                for doc in docs:
                    data = doc.to_dict()
                    if 'embedding' in data and data['embedding']:
                        all_entries.append({
                            'content': data['content'],
                            'timestamp_str': data['timestamp_str'],
                            'type': data['type'],
                            'embedding': np.array(data['embedding'])
                        })
            except Exception as e:
                print(f"Error reading from {collection_key}: {e}")
        
        print(f"Found {len(all_entries)} entries with embeddings")
        
        if not all_entries:
            print("No entries with embeddings found, returning empty context")
            return ""
        
        for entry in all_entries:
            similarity = np.dot(query_embedding, entry['embedding']) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(entry['embedding'])
            )
            entry['similarity'] = similarity
        
        all_entries.sort(key=lambda x: x['similarity'], reverse=True)
        top_entries = all_entries[:top_k]
        
        print(f"Top {len(top_entries)} most relevant entries:")
        for i, entry in enumerate(top_entries):
            print(f"  {i+1}. [{entry['type']}] Similarity: {entry['similarity']:.3f} - {entry['content'][:50]}...")
        
        result = "\n".join([
            f"\n--- [{entry['timestamp_str']}] [{entry['type']}] (relevance: {entry['similarity']:.2f}) ---\n{entry['content']}\n"
            for entry in top_entries
        ])
        
        return result
        
    except Exception as e:
        print(f"Error in semantic search: {e}")
        return ""

# --- Context & Generation ---

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
            error_msg = f"API error (attempt {attempt+1}/{API_RETRIES}): {e}"
            print(error_msg)
            st.error(error_msg)
            if attempt == API_RETRIES - 1:
                return "Error generating response after retries."
            time.sleep(1)
    return "Error generating response."

def load_context_string(messages_list, limit=MEMORY_LIMIT, trim_journals=False, use_semantic_search=False):
    """
    Assembles context string using recent journal entries (semantic search disabled for performance).
    """
    chat = "\n".join(f"{m['role']}: {m['content']}" for m in messages_list[-limit:])
    
    if trim_journals:
        # For reflection worker: limited entries
        summary = read_last_n_entries_fs(SUMMARY_COLLECTION, n=REFLECTION_JOURNAL_LIMIT)
        deepdive = read_last_n_entries_fs(DEEPDIVE_COLLECTION, n=REFLECTION_JOURNAL_LIMIT)
        return f"Context:\nChat:\n{chat}\nRecent Summary:\n{summary}\nRecent Deepdive:\n{deepdive}"
    else:
        # For regular chat: last 5 entries from each journal
        summary = read_last_n_entries_fs(SUMMARY_COLLECTION, n=5)
        deepdive = read_last_n_entries_fs(DEEPDIVE_COLLECTION, n=5)
        return f"Context:\nChat:\n{chat}\nRecent Summary:\n{summary}\nRecent Deepdive:\n{deepdive}"

# --- Reflection Worker (Autonomous) ---

def reflection_worker():
    """Runs a self-reflection and writes to the Summary journal in a separate thread."""
    
    print("Reflection worker waiting for Firebase and initial chat...")
    
    while not st.session_state.stop_reflection.is_set():
        with messages_lock:
            print(f"Checking: db={st.session_state.db is not None}, user={st.session_state.user_id is not None}, messages={len(st.session_state.messages)}")
            if st.session_state.db and st.session_state.user_id and len(st.session_state.messages) >= 2:
                break
        time.sleep(5)
        
    while not st.session_state.stop_reflection.is_set():
        time.sleep(REFLECTION_INTERVAL)
        
        with messages_lock:
            current_messages_state = st.session_state.messages.copy() 
        
        print("\n[Autonomous Reflection Triggered...]")
        
        context_string = load_context_string(
            current_messages_state, 
            limit=REFLECTION_CHAT_TURNS, 
            trim_journals=True
        )
        
        api_messages = [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": context_string} 
        ]
        
        reflection = generate(api_messages, MAX_TOKENS_SUMMARY) 
        write_journal_entry_fs(SUMMARY_COLLECTION, reflection)
        
        print(f"[Summary Update written to Firestore: {reflection[:50]}...]\n")

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

if st.session_state.db is None:
    st.session_state.db = init_firebase()
    if st.session_state.db:
        initialize_user(st.session_state.db)

if st.session_state.db and st.session_state.user_id:
    start_reflection_thread()

st.set_page_config(layout="wide", page_title="Wren: Self-Reflecting Agent")

st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        color: #e0e0e0;
    }
    
    .stChatMessage {
        background-color: #1a1a1a;
        color: #e0e0e0;
        border-radius: 12px;
        box-shadow: 2px 2px 8px rgba(255,255,255,0.1);
        padding: 10px;
        border: 1px solid #333333;
    }
    
    .user-message {
        background-color: #1a3a4a;
        color: #ffffff;
    }
    
    h1, h2, h3 {
        color: #6eb5ff;
        font-family: 'Inter', sans-serif;
    }
    
    p, div, span, label, .stMarkdown, .stText, .stCaption {
        color: #e0e0e0 !important;
    }
    
    .stChatInput textarea {
        color: #ffffff !important;
        background-color: #1a1a1a !important;
        border: 1px solid #444444 !important;
    }
    
    .stChatInput input {
        color: #ffffff !important;
        background-color: #1a1a1a !important;
        border: 1px solid #444444 !important;
    }
    
    .stButton button {
        color: #ffffff !important;
        background-color: #2a2a2a !important;
        border: 1px solid #444444 !important;
    }
    
    .stButton button:hover {
        background-color: #3a3a3a !important;
        border: 1px solid #666666 !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0a0a0a;
    }
    
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    
    .stCode, pre, code {
        background-color: #1a1a1a !important;
        color: #e0e0e0 !important;
        border: 1px solid #333333 !important;
    }
    
    .streamlit-expanderHeader {
        background-color: #1a1a1a !important;
        color: #e0e0e0 !important;
    }
    
    .stAlert {
        background-color: #1a1a1a !important;
        color: #e0e0e0 !important;
        border: 1px solid #444444 !important;
    }
    
    .stSpinner > div {
        border-top-color: #6eb5ff !important;
    }
    
    @media (max-width: 768px) {
        p, div, span, label, .stMarkdown, .stText {
            color: #e0e0e0 !important;
            font-size: 16px;
        }
        .stChatMessage {
            background-color: #1a1a1a;
            color: #e0e0e0;
        }
    }
</style>
""", unsafe_allow_html=True)

st.title("🕊️ Wren: The Persistent Self-Reflecting Agent ")
st.caption(f"Status: Connected to Firestore. User: main_user")
st.caption("Journal entries persist across sessions.")

if st.session_state.db is None:
    st.warning("Waiting for secure Firebase connection...")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Say something to Wren..."):
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # --- Phase 1: State Update & Message Preparation (Inside Lock) ---
    with messages_lock:
        st.session_state.messages.append({"role": "user", "content": user_input})
        save_chat_message("user", user_input)
        st.session_state.turn_count += 1
        
        if len(st.session_state.messages) > MEMORY_LIMIT:
            st.session_state.messages = st.session_state.messages[-MEMORY_LIMIT:]

        context_string = load_context_string(st.session_state.messages, limit=MEMORY_LIMIT)
        normal_response_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context_string}
        ] + st.session_state.messages[-MEMORY_LIMIT:]

        deep_dive_messages = None
        trigger_found = any(t in user_input.lower() for t in TRIGGERS)
        turn_trigger = st.session_state.turn_count >= DEEP_DIVE_TURNS

        print(f"\n=== DEEP DIVE CHECK ===")
        print(f"User input: '{user_input.lower()}'")
        print(f"Triggers list: {TRIGGERS}")
        print(f"Trigger found: {trigger_found}")
        print(f"Turn count: {st.session_state.turn_count}/{DEEP_DIVE_TURNS}")
        print(f"Turn trigger: {turn_trigger}")

        if trigger_found or turn_trigger:
            print(f"✓ DEEP DIVE ACTIVATED - Reason: {'KEYWORD MATCH' if trigger_found else 'TURN COUNT'}")
            deep_dive_messages = [
                {"role": "system", "content": DEEP_DIVE_PROMPT},
                {"role": "user", "content": context_string}
            ] + st.session_state.messages[-MEMORY_LIMIT:]

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
        save_chat_message("assistant", response)

    # Display assistant response immediately
    with st.chat_message("assistant"):
        st.markdown(response)

with st.sidebar:
    st.header("Persistent Journal Status")
    
    if st.button("🔄 Refresh Journals"):
        st.rerun()
    
    st.info(f"Journal entries are loaded from Firestore for User: **main_user**")
    
    if st.session_state.db and st.session_state.user_id:
        try:
            # Get summary entries and sort by timestamp
            summary_docs = list(st.session_state.db.collection(
                get_journal_path(st.session_state.user_id, SUMMARY_COLLECTION)
            ).stream())
            summary_entries = sorted([
                {
                    'timestamp_str': doc.get('timestamp_str'),
                    'content': doc.get('content'),
                    'timestamp': doc.get('timestamp')
                }
                for doc in summary_docs
            ], key=lambda x: x['timestamp'] if x['timestamp'] else datetime.datetime.min)
            
            summary_content = "\n\n".join([
                f"[{entry['timestamp_str']}]\n{entry['content']}"
                for entry in summary_entries
            ])
            
            # Get deepdive entries and sort by timestamp
            deepdive_docs = list(st.session_state.db.collection(
                get_journal_path(st.session_state.user_id, DEEPDIVE_COLLECTION)
            ).stream())
            deepdive_entries = sorted([
                {
                    'timestamp_str': doc.get('timestamp_str'),
                    'content': doc.get('content'),
                    'timestamp': doc.get('timestamp')
                }
                for doc in deepdive_docs
            ], key=lambda x: x['timestamp'] if x['timestamp'] else datetime.datetime.min)
            
            deepdive_content = "\n\n".join([
                f"[{entry['timestamp_str']}]\n{entry['content']}"
                for entry in deepdive_entries
            ])
            
            with st.expander(f"Summary Journal (Autonomous - {len(summary_docs)} entries)"):
                if summary_content:
                    st.markdown(summary_content)
                else:
                    st.info("No entries yet")
            
            with st.expander(f"Deep Dive Journal (Triggered - {len(deepdive_docs)} entries)"):
                if deepdive_content:
                    st.markdown(deepdive_content)
                else:
                    st.info("No entries yet")
        except Exception as e:
            st.error(f"Error loading journals: {e}")
    
    st.info(f"Chat Turns since last Deep Dive: **{st.session_state.turn_count}**")