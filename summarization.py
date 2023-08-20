from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import streamlit as st
import nltk
from PyPDF2 import PdfReader

possible_models = [
    "mabrouk/amazon-review-summarizer-bart",
    "MurkatG/review-summarizer-en",
]

def createModel(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model

def generateSummary(text, tokenizer, model):
    tokenized_text = tokenizer(text, return_tensors="pt")

    if len(tokenized_text["input_ids"][0]) > 1024:
        return "Text input is too long. Please shorten the text or upload as PDF."

    with torch.no_grad():
        output = model.generate(**tokenized_text)
            
    summary = tokenizer.decode(output[0], skip_special_tokens=True, max_new_tokens = 250, early_stopping=False)
    
    return summary

def convertPDFtoText(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    count = len(pdf_reader.pages)
    text = ""
    for i in range(count):
        page = pdf_reader.pages[i]
        text += page.extract_text()

    return text

def generateChunks(text, tokenizer):
    chunks = []
    current_chunk = ""
    
    sentences = nltk.sent_tokenize(text)

    for sentence in sentences:
        if(len(tokenizer.tokenize(current_chunk)) + len(tokenizer.tokenize(sentence)) < 1024):
            current_chunk += sentence
            current_chunk += " "
        else:
            chunks.append(current_chunk)
            current_chunk = ""
    
    return chunks

def main():
    st.set_page_config(page_title="Summarization App", layout="wide")
    
    model_name = "mabrouk/amazon-review-summarizer-bart"

    st.title("Text Summarization App")
    model_name = st.sidebar.selectbox("Select Model", possible_models)
    st.write("Using the " + model_name + " model")
    tokenizer, model = createModel(model_name)

    tab1, tab2, tab3 = st.tabs(["Text", "Document", "Corpus"])

    with tab1:
        st.header("Text Summarization")
        user_input = st.text_area("Enter your text here")

        if st.button("Generate Summary"):
            summary = generateSummary(user_input, tokenizer, model)
            st.subheader("Generated Summary: ")
            st.write(summary)
    
    with tab2:
        st.header("Document Summarization")
        uploaded_file = st.file_uploader("Upload PDF file", type="pdf")
        print(uploaded_file)

        if(uploaded_file is not None):
            st.subheader("Generated Summary for " + uploaded_file.name + ": ")

            #convert pdf to text            
            text = convertPDFtoText(uploaded_file)
            chunks = generateChunks(text, tokenizer)

            for chunk in chunks:
                summary = generateSummary(chunk, tokenizer, model)
                st.write(summary)
    with tab3:
        st.header("Corpus Summarization")

if __name__ == '__main__':
    main()