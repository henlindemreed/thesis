

import tensorflow as tf


def tf_word_list(str_tensor):
    return tf.strings.split(str_tensor)

def tf_num_words(str_tensor) -> int:
    return len(tf_word_list(str_tensor))

