import pika
import pandas as pd
import time
import sys

# RabbitMQ connection parameters
RABBITMQ_HOST = 'localhost'
QUEUE_NAME = 'progress_queue'

def send_progress(progress: float, message: str):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.basic_publish(
        exchange='',
        routing_key=QUEUE_NAME,
        body=f"{progress},{message}".encode()
    )
    connection.close()

def process_excel_file(file_path):
    excel_file = pd.ExcelFile(file_path)
    total_sheets = len(excel_file.sheet_names)
    for i, sheet_name in enumerate(excel_file.sheet_names):
        # Simulate processing time
        time.sleep(1)  # Simulate work
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        progress = ((i + 1) / total_sheets) * 100
        send_progress(progress, f"Processing sheet {sheet_name} ({i + 1}/{total_sheets})")
    send_progress(100, "Processing complete!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python producer.py <excel_file_path>")
        sys.exit(1)
    process_excel_file(sys.argv[1])
