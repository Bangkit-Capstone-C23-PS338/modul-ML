from transformers import TFBertForSequenceClassification
import tensorflow as tf
from transformers import BertTokenizer

# Load the model
model = TFBertForSequenceClassification.from_pretrained("model/nnmodel")
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
# Assuming you have loaded the model as 'model'

# Define your input text
input_text = "Kamu itu bodoh sekali hari ini"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors='tf')

# Make the prediction
outputs = model(inputs)

# Get the predicted probabilities for each class
probabilities = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]

# Get the predicted label
predicted_label = tf.argmax(probabilities).numpy()

# Print the predicted label and probabilities
print("Input Text:", input_text)
print("Predicted label:", predicted_label)
print("Probabilities:", probabilities)
print("Sentiment score:", probabilities[1])