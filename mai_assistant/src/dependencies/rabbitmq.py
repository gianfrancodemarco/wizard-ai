import pika

# Connection parameters
# connect with username and password
connection_params = pika.ConnectionParameters(host='localhost', port=5672, credentials=pika.PlainCredentials('user', 'password'))

# Establish a connection to the RabbitMQ server
connection = pika.BlockingConnection(connection_params)
channel = connection.channel()

# Declare a queue named 'hello' (create if not exists)
channel.queue_declare(queue='hello')

# Publish a message to the 'hello' queue
message_body = 'Hello, RabbitMQ!'
channel.basic_publish(exchange='', routing_key='hello', body=message_body)

print(f" [x] Sent '{message_body}'")

# Close the connection
connection.close()