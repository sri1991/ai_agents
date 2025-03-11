import streamlit as st
import logging
import time
from io import StringIO

# Custom Streamlit logging handler
class StreamlitHandler(logging.Handler):
    def __init__(self, update_logs_callback):
        super().__init__()
        self.update_logs_callback = update_logs_callback
        self.logs = []

    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(log_entry)
        self.update_logs_callback(self.logs)

# Configure logging
log_buffer = StringIO()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(log_buffer)]
)
logger = logging.getLogger(__name__)

# Streamlit app
st.title("Streamlit App with Real-Time Logs")

# Session state to store logs and task status
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'task_running' not in st.session_state:
    st.session_state.task_running = False

# Callback to update logs in session state
def update_logs(new_logs):
    st.session_state.logs = new_logs
    st.experimental_rerun()

# Add custom handler to logger
if not any(isinstance(handler, StreamlitHandler) for handler in logger.handlers):
    streamlit_handler = StreamlitHandler(update_logs)
    logger.addHandler(streamlit_handler)

# Display logs
log_display = st.text_area("Task Logs", "\n".join(st.session_state.logs), height=200)

# Simulate a task with logging
def run_task():
    st.session_state.task_running = True
    logger.info("Starting task...")
    
    items = ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5"]
    total_items = len(items)
    
    for i, item in enumerate(items):
        time.sleep(1)  # Simulate processing time
        progress = ((i + 1) / total_items) * 100
        logger.info(f"Processing {item} ({i + 1}/{total_items}) - {progress:.1f}%")
    
    logger.info("Task completed! 100%")
    st.session_state.task_running = False

# Button to start the task
if st.button("Start Task") and not st.session_state.task_running:
    st.session_state.logs = []  # Clear previous logs
    run_task()

# Progress bar based on log messages
if st.session_state.logs:
    latest_log = st.session_state.logs[-1]
    if "%" in latest_log:
        progress = float(latest_log.split("%")[0].split()[-1])
        st.progress(progress / 100)

# Stop button
if st.button("Stop Task") and st.session_state.task_running:
    st.session_state.task_running = False
    logger.warning("Task stopped by user!")
    st.experimental_rerun()
