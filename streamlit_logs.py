import streamlit as st
import time
import os

# File to store messages
LOG_FILE = "progress_log.txt"

# Clear the log file at the start
if not os.path.exists(LOG_FILE) or st.session_state.get('clear_log', True):
    with open(LOG_FILE, 'w') as f:
        f.write("")
    if 'clear_log' not in st.session_state:
        st.session_state.clear_log = False

# Session state to track task status and progress
if 'task_running' not in st.session_state:
    st.session_state.task_running = False
if 'progress' not in st.session_state:
    st.session_state.progress = 0

# Streamlit app
st.title("Simple Progress App with File-Based Messages")

# Function to write messages to the file
def write_message(message):
    with open(LOG_FILE, 'a') as f:
        f.write(message + "\n")

# Function to read messages from the file
def read_messages():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return []

# Display current messages and progress bar
messages = read_messages()
if messages:
    # Display messages
    st.text_area("Progress Messages", "\n".join(messages), height=200)

    # Parse progress from the last message
    last_message = messages[-1]
    if "%" in last_message:
        try:
            progress = float(last_message.split("%")[0].split()[-1])
            st.session_state.progress = progress
        except (ValueError, IndexError):
            pass

# Display progress bar
st.progress(st.session_state.progress / 100)

# Simulate a task with progress messages
def run_task():
    st.session_state.task_running = True
    st.session_state.progress = 0
    write_message("Task starting...")
    
    items = ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"]
    total_steps = len(items)

    for i, step in enumerate(items):
        if not st.session_state.task_running:
            break
        progress = ((i + 1) / total_steps) * 100
        message = f"Processing {step} ({i + 1}/{total_steps}) - {progress:.1f}%"
        write_message(message)
        st.session_state.progress = progress
        st.experimental_rerun()
        time.sleep(1)  # Simulate work

    if st.session_state.task_running:
        write_message("Task completed! 100%")
        st.session_state.task_running = False
        st.experimental_rerun()

# Button to start the task
if st.button("Start Task") and not st.session_state.task_running:
    # Clear the log file before starting
    with open(LOG_FILE, 'w') as f:
        f.write("")
    st.session_state.task_running = True
    run_task()

# Stop button
if st.button("Stop Task") and st.session_state.task_running:
    st.session_state.task_running = False
    write_message("Task stopped by user!")
    st.experimental_rerun()

# Reset button
if st.button("Reset"):
    st.session_state.progress = 0
    st.session_state.task_running = False
    with open(LOG_FILE, 'w') as f:
        f.write("Waiting to start...\n")
    st.experimental_rerun()
