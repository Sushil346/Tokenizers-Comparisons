import streamlit as st
from transformers import AutoTokenizer

st.title("Tokenizer Comparison App")
col1, col2 = st.columns(2)
with col1:
    model1 = st.selectbox("Model1",["gpt2", "bert-base-uncased", "t5-small","facebook/opt-125m"], key="m1")

with col2:
    model2= st.selectbox("Model2",["bert-base-uncased", "gpt2", "t5-small","facebook/opt-125m"], key="m2")

text_input = st.text_area("Enter text to tokenize:", "Hello, Wassup GenZ ?")

if st.button("Compare"):
    col1, col2 = st.columns(2)

    for col, model in zip([col1, col2], [model1, model2]):
        with col:
            st.subheader(model)
            tokenizer = AutoTokenizer.from_pretrained(model)

            tokens = tokenizer.tokenize(text_input)
            ids  = tokenizer.encode(text_input)
            decoded_text = tokenizer.decode(ids)

        
            st.write("  *Tokens Counts:* ", len(tokens))
            st.metric("Vocab Size",f"{tokenizer.vocab_size}")

            st.write(" **IDs:** ")
            st.code(str(ids))

            st.write(" **Tokens:** ")
            st.code(str(tokens))
            
            st.write(" **Decoded Text:** ")
            st.code(decoded_text)