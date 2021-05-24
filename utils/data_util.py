import pandas as pd
import tensorflow as tf
import tensorflow.keras as k
from sklearn.model_selection import train_test_split

#填充0，保持长度一致
def postpad_to(sequence, to):
    return k.preprocessing.sequence.pad_sequences(sequence, to, padding='post', truncating='post')

def load_data(file_name='../data/data.csv', split_ratio=None, random_state=42):
    data = pd.read_csv(file_name, index_col=False)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    if split_ratio is None:
        data_train = None
        data_test = None
    else:
        data_train, data_test = train_test_split(
            data,
            test_size=split_ratio,
            stratify=data['label'],
            shuffle=True,
            random_state=42
        )

    return data_train, data_test, data

def create_dataset_preloaded(char_vectorizer, word_vectorizer, data: pd.DataFrame, vec_length=200):
    assert word_vectorizer is not None or char_vectorizer is not None

    if char_vectorizer is not None:
        cv = tf.constant(postpad_to(char_vectorizer.texts_to_sequences(data['url']), vec_length), name='char')

    if word_vectorizer is not None:
        word_tokenizer = word_vectorizer.build_tokenizer()
        wv = tf.constant(postpad_to(data['url'].map(lambda url: [word_vectorizer.vocabulary_.get(a, -1)+2 for a in word_tokenizer(url)]), vec_length), name='word')

    targets = [1 if x=="bad" else 0 for x in data["label"]]

    if word_vectorizer is not None:
        if char_vectorizer is not None:
            ds = tf.data.Dataset.from_tensor_slices(((wv, cv), targets))
        else:
            ds = tf.data.Dataset.from_tensor_slices((wv, targets))
    else:
        ds = tf.data.Dataset.from_tensor_slices((cv, targets))

    return ds

def create_dataset_generator(char_vectorizer, word_vectorizer, data: pd.DataFrame, vec_length=200):
    assert char_vectorizer is not None or word_vectorizer is not None

    if word_vectorizer is not None:
        word_tokenizer = word_vectorizer.build_tokenizer()

    def gen():
        for row in data.iterrows():
            out_dict = dict()

            url = row[1].url
            _label = row[1].label
            if _label == "bad":
                target = 1
            else:
                target = 0

            if word_vectorizer is not None:
                wv = tf.constant(postpad_to(
                    [[word_vectorizer.vocabulary_.get(a, -1) + 2 for a in word_tokenizer(url)]]  # 0 = padding, 1 = OOV
                    , vec_length), name='word')
                out_dict['word'] = tf.squeeze(wv)

            if char_vectorizer is not None:
                cv = tf.constant(postpad_to(char_vectorizer.texts_to_sequences([url]), vec_length), name='char')
                out_dict['char'] = tf.squeeze(cv)

            yield out_dict, target

    output_types, output_shapes = dict(), dict()
    if word_vectorizer is not None:
        output_types['word'] = tf.float64
        output_shapes['word'] = tf.TensorShape([vec_length])
    if char_vectorizer is not None:
        output_types['char'] = tf.float64
        output_shapes['char'] = tf.TensorShape([vec_length])

    ds = tf.data.Dataset.from_generator(
        gen,
        output_types=(output_types, tf.int32),
        output_shapes=(output_shapes, tf.TensorShape([]))
    )

    return ds
