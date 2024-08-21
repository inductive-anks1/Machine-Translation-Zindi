import numpy as np
import yaml
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import mlflow
import mlflow.tensorflow
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Concatenate, Dropout, Input, Dense, 
                                     Add, LayerNormalization, MultiHeadAttention)
from transformers import FlaubertTokenizer, TFFlaubertModel
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, Callback
import pathlib

# Data loading functions
def load_data(data_path):
    return pd.read_csv(data_path)

def concat_dataframes(df1, df2):
    return pd.concat([df1, df2], ignore_index=True)

def concat_dataframes(df1, df2):
    return pd.concat([df1, df2], ignore_index=True)


def load_datasets(train_path, test_path, validate_path):
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    validate = pd.read_parquet(validate_path)
    return train, test, validate


def load_flaubert():
    flaubert_tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased')
    flaubert_model = TFFlaubertModel.from_pretrained('flaubert/flaubert_base_cased')
    return flaubert_tokenizer, flaubert_model


def encode_sentences(sentences, tokenizer, max_len):
    return tokenizer(
        sentences,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )


def create_transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    x_ffn = Dense(ff_dim, activation="relu")(x)
    x_ffn = Dropout(dropout)(x_ffn)
    x_ffn = Dense(inputs.shape[-1])(x_ffn)
    x_ffn = Add()([x_ffn, x])
    x_ffn = LayerNormalization(epsilon=1e-6)(x_ffn)
    return x_ffn


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


def build_model(max_dyu_len, max_fr_len, flaubert_model, flaubert_tokenizer, head_size = 1024,
                num_heads = 16, ff_dim = 4096, dropout = 0.1, num_layers = 5):
    
    flaubert_embedding_layer = FlauBERTEmbeddingLayer(flaubert_model)

    # head_size = 1024
    # num_heads = 16
    # ff_dim = 4096
    # dropout = 0.1
    # num_layers = 5

    encoder_inputs = Input(shape=(max_dyu_len,), name='encoder_inputs', dtype=tf.int32)
    encoder_embeddings = flaubert_embedding_layer(encoder_inputs)
    pos_encoding = PositionalEncoding(max_dyu_len, encoder_embeddings.shape[-1])
    encoder_embeddings = pos_encoding(encoder_embeddings)

    x = encoder_embeddings
    for _ in range(num_layers):
        x = create_transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    encoder_outputs = x

    decoder_inputs = Input(shape=(max_fr_len,), name='decoder_inputs', dtype=tf.int32)
    decoder_embeddings = flaubert_embedding_layer(decoder_inputs)
    decoder_embeddings = pos_encoding(decoder_embeddings)

    x = decoder_embeddings
    for _ in range(num_layers):
        x = create_transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    decoder_outputs = x

    output_layer = Dense(flaubert_tokenizer.vocab_size, activation='softmax', name='output_layer')(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], output_layer)
    return model


def compile_model(model):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.9
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


def prepare_data(train_encodings_dyu, train_encodings_fr, max_fr_len):
    decoder_input_data = np.array(train_encodings_fr['input_ids'][:, :-1])
    decoder_target_data = np.expand_dims(np.array(train_encodings_fr['input_ids'][:, 1:]), axis=-1)

    decoder_input_data = pad_sequences(decoder_input_data, maxlen=max_fr_len, padding='post')
    decoder_target_data = pad_sequences(decoder_target_data, maxlen=max_fr_len, padding='post')

    encoder_input_data = np.array(train_encodings_dyu['input_ids'])
    decoder_input_data = np.array(decoder_input_data)
    decoder_target_data = np.array(decoder_target_data)

    return encoder_input_data, decoder_input_data, decoder_target_data


def split_data(encoder_input_data, decoder_input_data, decoder_target_data, test_size=0.2):
    return train_test_split(
        encoder_input_data, decoder_input_data, decoder_target_data, test_size=test_size
    )


class StopAtAccuracy(Callback):
    def __init__(self, accuracy_threshold=0.98):
        super(StopAtAccuracy, self).__init__()
        self.accuracy_threshold = accuracy_threshold

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') >= self.accuracy_threshold:
            print(f"\nReached {self.accuracy_threshold * 100:.2f}% accuracy. Stopping training.")
            self.model.stop_training = True


def train_model(model, encoder_input_train, decoder_input_train, decoder_target_train,
                encoder_input_val, decoder_input_val, decoder_target_val,
                batch_size=30, epochs=20):
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=1,
        restore_best_weights=True
    )

    stop_at_98 = StopAtAccuracy(accuracy_threshold=0.98)

    history = model.fit(
        [encoder_input_train, decoder_input_train],
        decoder_target_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([encoder_input_val, decoder_input_val], decoder_target_val),
        callbacks=[early_stopping, stop_at_98]
    )

    return history

def save_model(model, output_path):
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        model.save(output_path + '/transformer_translation_model.keras')



def main():
    params_file = 'params.yaml'
    params = yaml.safe_load(open(params_file))["train_model"]

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    train_path = home_dir.as_posix() + '/data/processed/train.csv'
    validate_path = home_dir.as_posix() + '/data/processed/validate.csv'
    test_path = home_dir.as_posix() + '/data/processed/test.csv'

    output_path =  home_dir.as_posix() + '/models'

    # Load data
    train = load_data(train_path)
    validate = load_data(validate_path)
    test = load_data(test_path)

    train = concat_dataframes(train, validate)

    train['dyu'] = train['dyu'].astype(str)
    train['fr'] = train['fr'].astype(str)

    # Load Flaubert and tokenize
    flaubert_tokenizer, flaubert_model = load_flaubert()

    # Tokenize data
    max_dyu_len = train['dyu_length'].max()
    max_fr_len = train['fr_length'].max() + 1 
    train_encodings_dyu = encode_sentences(train['dyu'].tolist(), flaubert_tokenizer, max_dyu_len)
    train_encodings_fr = encode_sentences(train['fr'].tolist(), flaubert_tokenizer, max_fr_len)

    # Start an MLflow experiment
    mlflow.set_experiment("translation_experiment")
    with mlflow.start_run():
        # Log the parameters
        mlflow.log_params(params)

        # Build and compile the model
        model = build_model(max_dyu_len, max_fr_len, flaubert_model, flaubert_tokenizer, 
                            head_size=params['head_size'], num_heads=params['num_heads'], 
                            ff_dim=params['ff_dim'], dropout=params['dropout'], num_layers=params['num_layers'])
        compile_model(model)

        # Prepare training data
        encoder_input_data, decoder_input_data, decoder_target_data = prepare_data(
            train_encodings_dyu, train_encodings_fr, max_fr_len
        )

        # Split data
        encoder_input_train, encoder_input_val, decoder_input_train, decoder_input_val, \
        decoder_target_train, decoder_target_val = split_data(
            encoder_input_data, decoder_input_data, decoder_target_data
        )

        # Train model
        history = train_model(
            model, encoder_input_train, decoder_input_train, decoder_target_train,
            encoder_input_val, decoder_input_val, decoder_target_val, 
            batch_size=params['batch_size'], epochs=params['epochs']
        )

        # Log metrics
        for epoch, acc in enumerate(history.history['accuracy']):
            mlflow.log_metric("train_accuracy", acc, step=epoch)
        for epoch, val_acc in enumerate(history.history['val_accuracy']):
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        for epoch, loss in enumerate(history.history['loss']):
            mlflow.log_metric("train_loss", loss, step=epoch)
        for epoch, val_loss in enumerate(history.history['val_loss']):
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        # Save model
        save_model(model, output_path + "/transformer_translation_model.keras")
        
        # Log the model
        mlflow.tensorflow.log_model(model, "model")

if __name__ == "__main__":
    main()
