import streamlit as st
import tensorflow as tf
from transformers import FlaubertTokenizer, TFFlaubertModel
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer

# Define FlauBERTEmbeddingLayer
class FlauBERTEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, flaubert_model, **kwargs):
        super(FlauBERTEmbeddingLayer, self).__init__(**kwargs)
        self.flaubert_model = flaubert_model

    def call(self, inputs):
        return self.flaubert_model(inputs)[0]

    def get_config(self):
        config = super().get_config()
        config.update({
            'flaubert_model': self.flaubert_model.name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        flaubert_model = TFFlaubertModel.from_pretrained(config['flaubert_model'])
        return cls(flaubert_model=flaubert_model)

# Define PositionalEncoding
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_seq_len, dm):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = self.positional_encoding_matrix(max_seq_len, dm)

    def positional_encoding_matrix(self, max_seq_len, dm):
        angle_rads = self.get_angles(np.arange(max_seq_len)[:, np.newaxis],
                                     np.arange(dm)[np.newaxis, :],
                                     dm)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        positional_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(positional_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, dm):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(dm))
        return pos * angle_rates

    def call(self, inputs):
        return inputs + self.positional_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_seq_len": self.positional_encoding.shape[1],
            "dm": self.positional_encoding.shape[2]
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(config['max_seq_len'], config['dm'])

# Load Flaubert model and tokenizer
flaubert_tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased')
flaubert_model = TFFlaubertModel.from_pretrained('flaubert/flaubert_base_cased')

# Load the trained model
model = tf.keras.models.load_model('/home/inductive-anks/zindi/machine-translation/Machine-Translation-Zindi/models/transformer_translation_model.keras/transformer_translation_model.keras',
                                   custom_objects={'FlauBERTEmbeddingLayer': FlauBERTEmbeddingLayer,
                                                   'PositionalEncoding': PositionalEncoding})

# Function to preprocess the input text
def preprocess_text(dyu_text, max_len):
    dyu_text = dyu_text.lower()
    dyu_encoding = flaubert_tokenizer(
        dyu_text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
    return dyu_encoding

# Streamlit app
st.title("Dyula to French Translation")

# Text input from user
dyu_input = st.text_area("Enter a sentence in Dyula:")

# Predict button
if st.button("Translate"):
    if dyu_input:
        # Preprocess the input
        max_dyu_len = model.input[0].shape[1]
        max_fr_len = model.input[1].shape[1]
        
        dyu_encoding = preprocess_text(dyu_input, max_dyu_len)

        # Generate dummy decoder input for the start of the sequence
        decoder_input = np.zeros((1, max_fr_len), dtype=int)

        # Predict the French sentence
        predicted = model.predict([dyu_encoding['input_ids'], decoder_input])

        # Convert the predicted token IDs to a French sentence
        predicted_ids = np.argmax(predicted, axis=-1)
        predicted_sentence = flaubert_tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

        # Display the predicted sentence
        st.write(f"Predicted French sentence: {predicted_sentence}")
    else:
        st.write("Please enter a sentence in Dyula.")
