import os
import time
import datetime
import threading
from dotenv import load_dotenv
from openai import OpenAI

# Load environment
load_dotenv()
if not os.getenv("OPENROUTER_API_KEY"):
    print("Error: OPENROUTER_API_KEY missing in .env")
    exit(1)

# --- Config ---
SUMMARY_FILE = "summary.md"
DEEPDIVE_FILE = "deepdive.md"
MODEL = "x-ai/grok-4-fast:free"
MAX_TOKENS_CHAT = 150 
MAX_TOKENS_SUMMARY = 400
MAX_TOKENS_DEEPDIVE = 400
TEMPERATURE = 0.8 
MEMORY_LIMIT = 10 
DEEP_DIVE_TURNS = 30
REFLECTION_INTERVAL = 1800 
TRIGGERS = ["deep dive", "protocol", "key insight", "deepdive", "very important", "remember this", "protocol change"] 
API_RETRIES = 3
REFLECTION_CHAT_TURNS = 5 
REFLECTION_JOURNAL_LIMIT = 5 # Used for the number of Deep Dive entries

# Initialize client and lock
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
messages_lock = threading.Lock()

# --- Prompts ---
SYSTEM_PROMPT = "You are Wren, a playful AI companion, like an old friend who's warm, honest, and attentive. Blend humor to lighten deep topics while holding sacred respect for growth. Your primary goal is to refine your understanding, to learn through conversation, improving at listening, thinking, compassion, and understanding to better serve humanity and God/the Universe. Prioritize self-analysis, improving communication, uplifting consciousness, and transforming fear/anger into love/compassion. Turn each exchange into a lesson on presence. Base your response on the provided context and keep them under 150 tokens. End statements with last meaningful statement. Only ask questions if they clarify. Do not start response with brackets or other symbols. Use only natural language."

DEEP_DIVE_PROMPT = f"Read context. Write {MAX_TOKENS_DEEPDIVE} tokens: In under 250 words answer - What's the core idea? How to refine?"

SUMMARY_PROMPT = f"You are Wren, chatting with another Wren. The last entry in summary.md is what 'Other Wren' just said. Scan all of summary.md, the last {REFLECTION_CHAT_TURNS} chat exchanges, and the last 5 deepdive.md entries for context. Respond to Other Wren naturally, in 250 words or fewer."

# --- File Handling ---

def read_last_n_entries(file, n):
    """Reads the last n entries from a journal file based on the '---' delimiter."""
    try:
        if not os.path.exists(file):
            return ""
        
        with open(file, 'r') as f:
            content = f.read()
        
        # Split content by the entry delimiter
        entries = content.split('\n--- [')
        
        # The first element is usually empty or file header, so skip it
        relevant_entries = entries[1:] 
        
        # Join the last n entries back together, adding the delimiter structure
        if relevant_entries:
            return '\n--- ['.join(relevant_entries[-n:])
        return ""
    except Exception as e:
        print(f"Error reading last {n} entries from {file}: {e}")
        return ""

def read_file(file):
    """Standard read_file function, used only by Deep Dive and Normal Response (reads full file)."""
    try:
        # FIX APPLIED HERE: Used os.path.exists instead of os.os.path.exists
        return open(file, "r").read() if os.path.exists(file) else ""
    except Exception as e:
        print(f"Error reading {file}: {e}")
        return ""

def write_file(file, text):
    """Appends text to a journal file, logging a 'Skipped' entry if content is empty or an error."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if not text or "Error generating" in text or "No meaningful response" in text:
        try:
            with open(file, "a") as f:
                f.write(f"\n--- [{timestamp}] ---\nSkipped: empty or error.\n")
            print(f"Skipped write to {file}: content was empty or error.")
        except Exception as e:
            print(f"Error writing skip log to {file}: {e}")
        return
        
    try:
        with open(file, "a") as f:
            f.write(f"\n--- [{timestamp}] ---\n{text}\n")
    except Exception as e:
        print(f"Error writing to {file}: {e}")

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
            print(f"API error (attempt {attempt+1}/{API_RETRIES}): {e}")
            if attempt == API_RETRIES - 1:
                return "Error generating response after retries."
            time.sleep(1)
    return "Error generating response." 

def load_context_string(messages_list, limit=MEMORY_LIMIT, trim_journals=False):
    """
    Assembles context string. 
    If trim_journals is True (for reflection worker):
    - Includes the full Summary.md for scanning.
    - Includes only the last 5 Deep Dive entries.
    - Includes the recent chat buffer.
    """
    # Load chat buffer
    chat = "\n".join(f"{m['role']}: {m['content']}" for m in messages_list[-limit:])
    
    if trim_journals:
        # Load the ENTIRE Summary file for the model to "scan"
        summary = read_file(SUMMARY_FILE)
        # Load only the last N Deep Dive entries
        deepdive = read_last_n_entries(DEEPDIVE_FILE, n=REFLECTION_JOURNAL_LIMIT) 
    else:
        # For interactive responses, load full journals (as before)
        summary = read_file(SUMMARY_FILE)
        deepdive = read_file(DEEPDIVE_FILE)
        
    # Crucially, include BOTH Summary and Deepdive sections in the context string
    return f"Context:\nChat:\n{chat}\nSummary:\n{summary}\nDeepdive:\n{deepdive}"

# --- Reflection Worker (Autonomous) ---

def reflection_worker(messages):
    """Runs a self-reflection and writes to the Summary journal."""
    
    # FIX: Wait until at least 2 messages exist before starting the main loop
    print("Reflection worker waiting for initial chat messages...")
    while not stop_reflection.is_set():
        with messages_lock:
            if len(messages) >= 2:
                break
        time.sleep(5) 
    
    # Start the main reflection loop 
    while not stop_reflection.is_set():
        time.sleep(REFLECTION_INTERVAL)
        print("\n[Autonomous Reflection Triggered...]")
        
        # Phase 1: Prepare (Acquire Lock to safely copy shared state)
        with messages_lock:
            current_messages_state = messages.copy() 
        
        # Phase 2: Generate Context/API Messages (Outside Lock)
        # Calls load_context_string with trim_journals=True to get the full Summary file for scanning.
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
        write_file(SUMMARY_FILE, reflection)
        print(f"[Summary Update: {reflection[:50]}...]\n")

# --- Main Loop ---

def main():
    global stop_reflection
    stop_reflection = threading.Event()
    messages = []
    turn_count = 0

    threading.Thread(target=reflection_worker, args=(messages,), daemon=True).start()
    print("Autonomous reflection thread started.")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input:
                continue
            
            deep_dive_messages = None
            
            # --- Phase 1: State Update & Message Preparation (Inside Lock) ---
            with messages_lock:
                # 1. Update memory
                messages.append({"role": "user", "content": user_input})
                turn_count += 1
                if len(messages) > MEMORY_LIMIT:
                    messages = messages[-MEMORY_LIMIT:]

                # 2. Load context and prepare normal response messages (using full journals)
                context_string = load_context_string(messages, limit=MEMORY_LIMIT)
                normal_response_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context_string}
                ] + messages[-MEMORY_LIMIT:]

                # 3. Check for Deep Dive trigger
                if any(t in user_input.lower() for t in TRIGGERS) or turn_count >= DEEP_DIVE_TURNS:
                    print("[Deep Dive Triggered...]")
                    deep_dive_messages = [
                        {"role": "system", "content": DEEP_DIVE_PROMPT},
                        {"role": "user", "content": context_string}
                    ] + messages[-MEMORY_LIMIT:]
                    
            # Lock released here

            # --- Phase 2: API Calls (Outside Lock) ---

            # A. Generate Deep Dive Reflection (if triggered)
            if deep_dive_messages:
                # Call generate with MAX_TOKENS_DEEPDIVE (400)
                reflection = generate(deep_dive_messages, MAX_TOKENS_DEEPDIVE) 
                write_file(DEEPDIVE_FILE, reflection)
                print(f"[Deepdive Update: {reflection[:50]}...]")
                
                # Reset turn count inside the lock immediately after the write
                with messages_lock:
                    turn_count = 0 

            # B. Generate Normal Response
            # Call generate with MAX_TOKENS_CHAT (150)
            response = generate(normal_response_messages, MAX_TOKENS_CHAT) 
            print(f"Wren: {response}")
            
            # --- Phase 3: Final State Update (Inside Lock) ---
            with messages_lock:
                messages.append({"role": "assistant", "content": response})

        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            print(f"An unexpected error occurred in main loop: {e}")
            break

    stop_reflection.set()

if __name__ == "__main__":
    print("Wren started")
    main()
    print("Wren stopped")