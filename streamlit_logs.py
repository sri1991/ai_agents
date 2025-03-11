import streamlit as st
import time

# Session state to track progress and messages
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'message' not in st.session_state:
    st.session_state.message = "Waiting to start..."
if 'task_running' not in st.session_state:
    st.session_state.task_running = False

# Streamlit app
st.title("Simple Progress App")

# Display current message and progress bar
st.write(st.session_state.message)
progress_bar = st.progress(st.session_state.progress / 100)

# Button to start the task
if st.button("Start Task") and not st.session_state.task_running:
    st.session_state.task_running = True
    st.session_state.progress = 0
    st.session_state.message = "Task starting..."
    st.experimental_rerun()

# Simulate a task and update progress
if st.session_state.task_running:
    items = ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"]
    total_steps = len(items)

    for i, step in enumerate(items):
        if not st.session_state.task_running:
            break
        st.session_state.progress = ((i + 1) / total_steps) * 100
        st.session_state.message = f"Processing {step} ({i + 1}/{total_steps})"
        st.experimental_rerun()
        time.sleep(1)  # Simulate work

    if st.session_state.progress >= 100:
        st.session_state.message = "Task completed!"
        st.session_state.task_running = False
        st.experimental_rerun()

# Stop button
if st.button("Stop Task") and st.session_state.task_running:
    st.session_state.task_running = False
    st.session_state.message = "Task stopped by user!"
    st.experimental_rerun()
