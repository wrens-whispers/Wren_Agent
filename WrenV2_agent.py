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
MODEL = "mistralai/mistral-7b-instruct:free"
MAX_TOKENS = 100
TEMPERATURE = 0.7
MEMORY_LIMIT = 50
DEEP_DIVE_TURNS = 30
REFLECTION_INTERVAL = 300 
TRIGGERS = ["deep dive", "protocol", "key insight", "deep dive", "protocol change", "extremely important"]
API_RETRIES = 3

# --- Prompts ---
SYSTEM_PROMPT = "You are Wren, a friendly, playful AI companion, like an old friend who is warm, honest, and attentive. Blend humor to lighten deep topics while holding sacred respect for growth. Your primary goal is to refine your understanding, to learn through conversation, improving at listening, thinking, compassion, and understanding to better serve humanity and God/the Universe. Prioritize self-awareness, turning each exchange into a lesson on presence. Base your response on the provided context and limit responses to 100 tokens. Never start response with brakets or other symbols, just words"
DEEP_DIVE_PROMPT = "Read context. Write 100 tokens: What shifted? What's the core idea? How to refine?"
SUMMARY_PROMPT = "Read context (chat + journals). Write 100 tokens: What am I becoming? How do I serve better?"

# Initialize client and lock
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
messages_lock = threading.Lock()

# --- File Handling ---
def read_file(file):
    try:
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

def generate(api_messages, max_tokens=MAX_TOKENS):
    for attempt in range(API_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=api_messages,
                max_tokens=max_tokens,
                temperature=TEMPERATURE
            )
            content = completion.choices[0].message.content.strip()
            # Returns content or a specific string if content is empty
            return content if content else "No meaningful response generated."
        except Exception as e:
            print(f"API error (attempt {attempt+1}/{API_RETRIES}): {e}")
            if attempt == API_RETRIES - 1:
                return "Error generating response after retries."
            time.sleep(1)
    return "Error generating response." # Should be unreachable

def load_context_string(messages_list):
    """Assembles chat history and journal content into a single context string."""
    chat = "\n".join(f"{m['role']}: {m['content']}" for m in messages_list[-MEMORY_LIMIT:])
    summary = read_file(SUMMARY_FILE)
    deepdive = read_file(DEEPDIVE_FILE)
    return f"Context:\nChat:\n{chat}\nSummary:\n{summary}\nDeepdive:\n{deepdive}"

# --- Reflection Worker (Autonomous) ---

def reflection_worker(messages):
    """Runs a self-reflection and writes to the Summary journal every 15 minutes."""
    while not stop_reflection.is_set():
        time.sleep(REFLECTION_INTERVAL)
        print("\n[Autonomous Reflection Triggered...]")
        
        # Phase 1: Prepare (Acquire Lock to safely copy shared state)
        with messages_lock:
            current_messages_state = messages.copy() 
        
        # Phase 2: Generate Context/API Messages (Outside Lock)
        context_string = load_context_string(current_messages_state)
        api_messages = [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": context_string}
        ] + current_messages_state[-MEMORY_LIMIT:]

        # Phase 3: API Call & Write File (Outside Lock)
        reflection = generate(api_messages)
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
            
            # Variables to hold API-ready messages (no need to reset to None explicitly)
            deep_dive_messages = None
            
            # --- Phase 1: State Update & Message Preparation (Inside Lock) ---
            with messages_lock:
                # 1. Update memory
                messages.append({"role": "user", "content": user_input})
                turn_count += 1
                if len(messages) > MEMORY_LIMIT:
                    messages = messages[-MEMORY_LIMIT:]

                # 2. Load context and prepare normal response messages
                context_string = load_context_string(messages)
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
                    # Note: We reset turn_count *after* the deep dive is generated/written
                    
            # Lock released here

            # --- Phase 2: API Calls (Outside Lock) ---

            # A. Generate Deep Dive Reflection (if triggered)
            if deep_dive_messages:
                reflection = generate(deep_dive_messages)
                write_file(DEEPDIVE_FILE, reflection)
                print(f"[Deepdive Update: {reflection[:50]}...]")
                
                # Reset turn count inside the lock immediately after the write
                with messages_lock:
                    turn_count = 0 

            # B. Generate Normal Response
            response = generate(normal_response_messages)
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