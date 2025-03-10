import streamlit as st
import pika
import pandas as pd
import threading
import queue
from io import BytesIO

# Global queue to receive messages from RabbitMQ
message_queue = queue.Queue()

# RabbitMQ connection parameters
RABBITMQ_HOST = 'localhost'
QUEUE_NAME = 'progress_queue'

# Callback to handle RabbitMQ messages
def callback(ch, method, properties, body):
    progress, message = body.decode().split(',', 1)
    message_queue.put((float(progress), message))

# Start RabbitMQ consumer in a separate thread
def start_rabbitmq_consumer():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback, auto_ack=True)
    channel.start_consuming()

# Initialize Streamlit app
st.title("Excel Sheet Viewer with RabbitMQ Progress")

# File upload widget
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

# Start RabbitMQ consumer thread
if 'consumer_thread' not in st.session_state:
    consumer_thread = threading.Thread(target=start_rabbitmq_consumer, daemon=True)
    consumer_thread.start()
    st.session_state.consumer_thread = consumer_thread

# Session state for progress and data
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'progress_message' not in st.session_state:
    st.session_state.progress_message = "Waiting to process..."
if 'excel_file' not in st.session_state:
    st.session_state.excel_file = None

# Update progress from RabbitMQ messages
if not message_queue.empty():
    progress, message = message_queue.get()
    st.session_state.progress = progress
    st.session_state.progress_message = message
    st.experimental_rerun()

# Display progress bar and message
st.write(st.session_state.progress_message)
progress_bar = st.progress(st.session_state.progress / 100)

# Trigger processing when file is uploaded
if uploaded_file is not None and st.session_state.progress < 100:
    if st.button("Start Processing"):
        # Save uploaded file temporarily
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Run producer in a separate process
        import subprocess
        subprocess.Popen(['python', 'producer.py', file_path])
        st.session_state.progress_message = "Processing started..."

# Display sheets after processing
if st.session_state.progress == 100 and st.session_state.excel_file:
    excel_file = st.session_state.excel_file
    sheet_names = excel_file.sheet_names
    tabs = st.tabs(sheet_names)

    for sheet_name, tab in zip(sheet_names, tabs):
        with tab:
            st.subheader(f"Sheet: {sheet_name}")
            df = pd.read_excel(BytesIO(uploaded_file.getbuffer()), sheet_name=sheet_name)
            st.dataframe(df, use_container_width=True)

else:
    st.info("Please upload an Excel file to view its sheets.")
