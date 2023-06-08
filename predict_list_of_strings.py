from transformers import TFBertForSequenceClassification
import tensorflow as tf
from transformers import BertTokenizer

# Load the model
model = TFBertForSequenceClassification.from_pretrained("model/nnmodel")
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')

# Define your input text as a list of strings
input_texts = ["Kamu itu bodoh sekali hari ini", "Aku senang bertemu denganmu"]

# Tokenize the input texts
inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors='tf')

# Make the prediction
outputs = model(inputs)

# Get the predicted probabilities for each class
probabilities = tf.nn.softmax(outputs.logits, axis=1).numpy()

# Get the predicted labels
predicted_labels = tf.argmax(probabilities, axis=1).numpy()

# Print the predicted labels and probabilities for each input text
for i, input_text in enumerate(input_texts):
    print("Input Text:", input_text)
    print("Predicted label:", predicted_labels[i])
    print("Sentiment score:", probabilities[i][1])
    print()