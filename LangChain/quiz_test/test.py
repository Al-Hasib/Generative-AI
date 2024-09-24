import streamlit as st
from bs4 import BeautifulSoup
import PyPDF2

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    pdf_text = ""
    
    # Extract text from each page
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        pdf_text += page.extract_text()
        
    return pdf_text

# Function to extract text from HTML
def extract_text_from_html(html_file):
    file_content = html_file.read().decode("utf-8")
    soup = BeautifulSoup(file_content, "html.parser")
    
    # Extract the title of the HTML page
    title = soup.title.string if soup.title else "No title found"
    
    # Extract all the paragraphs <p> from the HTML
    paragraphs = soup.find_all("p")
    paragraph_text = "\n".join([para.get_text() for para in paragraphs])
    
    # Combine title and paragraphs into a single string
    full_text = f"Title: {title}\n\n{paragraph_text}"
    return full_text

# Function to extract text from plain text file
def extract_text_from_txt(txt_file):
    return txt_file.read().decode("utf-8")

# Streamlit app to upload and display file content
uploaded_file = st.file_uploader("Upload a PDF, Text, or HTML file", type=["pdf", "txt", "html"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1]  # Get the file extension
    
    # Display appropriate content based on file type
    if file_type == 'pdf':
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.write("Extracted Text from PDF:")
        st.text(pdf_text)
    
    elif file_type == 'txt':
        text_content = extract_text_from_txt(uploaded_file)
        st.write("Text File Content:")
        st.text(text_content)
    
    elif file_type == 'html':
        html_content = extract_text_from_html(uploaded_file)
        st.write("HTML File Content:")
        st.text(html_content)
