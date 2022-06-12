import streamlit as st
from tokenizers import Tokenizer
import fasttext
import ml_collections


def config():
    cfg_dictionary = {
        "model_file": "fasttext-model.ftz",
        "tokenizer": "tokenizer.json",
    }
    configuration = ml_collections.FrozenConfigDict(cfg_dictionary)

    return configuration


def predict(text, tokenizer, model):
    """
    Returns Prediction of FastText model.
    The input string is tokenized and is then passed to the model for prediction
    Args:
        text(str) : Input string
        tokenizer : Custom tokenizer object
        model : FastText model
    """
    tokenized_text = ' '.join(tokenizer.encode(text).tokens)
    return model.predict(tokenized_text)


if __name__ == '__main__':
    cfg = config()
    loaded_tokenizer = Tokenizer.from_file(cfg.tokenizer)
    loaded_model = fasttext.load_model(cfg.model_file)

    st.write("# Browse Node ID Classification")
    description = """ 
    Browse node ID's are numeric codes that identify inside Amazon, a given
    product category. There are more than 30 thousand product categories on 
    Amazon, each one identified by a unique Node ID. In Amazon's own words
    > *Browse Node ID's are positive integers that uniquely identify product
    > sets, such as Literature & Fiction: (17), Medicine: (13996), Mystery &
    > Thrillers: (18), Nonfiction: (53), Outdoors & Nature: (290060). Amazon
    > uses thousands of browse node ID's*
    """
    st.write(description)
    product_description = st.text_area("Enter the Product Description",
                                       height=400)
    node_id, precision = predict(product_description, loaded_tokenizer,
                                 loaded_model)
    browse_node_id, = node_id  # unpacking node_id
    st.write(f"# Node ID {browse_node_id[9:]}")
