import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from bs4 import BeautifulSoup
import PyPDF2


load_dotenv()

# model = ChatOpenAI(model = "gpt-4o")

# Access the API key from the environment
api_key = os.getenv("GOOGLE_GEN_API")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=api_key)


quiz_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Generate {num_questions} quiz questions with four multiple-choice options for each question. Highlight the correct answer."),
        ("human", "You have to following the below content to create the quizes\n \
         Content: {text}"),
    ]
)


# Function to extract text from PDF
def load_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    pdf_text = ""
    
    # Extract text from each page
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        pdf_text += page.extract_text()
        
    return pdf_text

# Function to extract text from HTML
def load_html(html_file):
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
def load_text(txt_file):
    return txt_file.read().decode("utf-8")



# Function to generate quiz using LangChain
def generate_quiz(num_questions, text):
    Chain = quiz_prompt | llm | StrOutputParser()
    result = Chain.invoke({"num_questions":num_questions, "text":text})
    return result

# Streamlit UI
st.title("Quiz Generator from PDF/Text/HTML using LangChain")

# File uploader
uploaded_file = st.file_uploader("Upload a file (PDF, Text, or HTML)", type=["pdf", "txt", "html"])
print(uploaded_file)

# Number of quiz questions input
num_questions = st.number_input("How many quiz questions would you like to generate?", min_value=1, max_value=20, value=5)

if uploaded_file is not None:
    # Determine file type and load the text
    file_type = uploaded_file.type
    file_name = uploaded_file.name

    if file_type == "application/pdf":
        file_text = load_pdf(uploaded_file)
    elif file_type == "text/plain":
        file_text = load_text(uploaded_file)
    elif file_type == "text/html":
        file_text = load_html(uploaded_file)
    else:
        st.error("Unsupported file format.")
        file_text = None

    # Display extracted text
    if file_text:
        # st.write("### Extracted Text:")
        # st.write(file_text[:1000] + "...")  # Display first 1000 characters of the text

        # Generate Quiz Button
        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz..."):
                quiz = generate_quiz(num_questions, file_text)
                st.write("### Generated Quiz:")
                st.write(quiz)
