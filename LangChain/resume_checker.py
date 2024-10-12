import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import PyPDF2
from langchain_core.messages import AIMessage
import fitz
#pip install pymupdf
import streamlit as st

load_dotenv()

api_key = os.getenv("GOOGLE_GEN_API")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=api_key)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You will compare the Resume and Job Description and provide the Output in a JSON format"),
    ("human", "You will take the resume and Job description: \n \
     Resume: {resume} \n \
     Job Description: {Job_description} \n \
     Based on the Job description, does the resume is perfect for the JOB and What's the percentage of Matching the resume against the JOB description. \
     The JSON Output Key will be Name of resume holder, email, is_perfect,is_okay, Matching Score in percentage")
])

# Function to extract text from PDF
def load_pdf(pdf_file):
    pdf_document = fitz.open(pdf_file)  # Open the PDF
    pdf_text_with_links = ""

    # Iterate through each page
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)  # Load individual page
        pdf_text_with_links += page.get_text("text")  # Extract text

        # Extract links from the page
        links = page.get_links()
        for link in links:
            if 'uri' in link:  # Check if it's a URI (URL)
                pdf_text_with_links += f"\n(Link: {link['uri']})"
    
    return pdf_text_with_links
def parser(aimessage:AIMessage)->str:
    return aimessage.content


# Function to generate quiz using LangChain
def resume_checker(resume, job_description):
    Chain = prompt_template | llm | parser
    result = Chain.invoke({"resume":resume, "Job_description":job_description})
    return result

resume = load_pdf(r"C:\Users\abdullah\projects\Langchain\Generative-AI\LangChain\RAG\data\My_CV_2.pdf")
# print(resume)

Job_description = """ Job Responsibilities

 Leverage cutting-edge AI/ML methods to develop AI/ML-powered solutions that align with business objectives.
 Prepare dataset/data-pipeline to feed into the machine learning models.
 Optimize ML applications to improve performance, speed, and scalability
 Work with internal stakeholders, analysts, engineers to explore and understand business data.
 Conduct experiments and test different technical approaches
 Collaborate with the data engineering team to ensure clean records
 Implement machine learning model lifecycle management framework.
 Build/maintain model performance tracking tools, to monitor model performance, identify drivers and provide recommendations for optimization
 Author technical documentation and reports to communicate process and results

Knowledge And Experience

Minimum qualifications:

 1 to 4 years of hands-on experience in machine learning models, applications, and pipelines. 
 Experience in developing, testing, debugging, maintaining, or launching ML products, as well as software design and architecture. 
 Proficiency in one or more programming languages (preferably Python). 
 Working knowledge of one or more SQL languages (preferably Snowflake, PostgreSQL, MySQL). 
 Knowledge of common machine learning and statistical packages, frameworks, and concepts. 
 Experience/knowledge of working with large data sets, distributed computing, and cloud computing platforms. 
 Experience using CI/CD tools and version control systems (e.g., Git). 
 Excellent written and verbal communication skills in English. 

Preferred qualifications:

 Strong understanding of machine learning algorithms and their practical implementation. 
 Excellent problem-solving skills and the ability to translate business requirements into technical solutions. 
 Experience deploying machine learning models in production environments. 
 Excellent analytical and problem-solving skills with keen attention to detail. 
 A team-oriented mindset, believing that AI/ML is a collaborative effort. 

Education

Bachelor’s degree in Computer Science, Mathematics, Statistics, Physics, or a related field with project experience in machine learning and statistical modeling.

Competencies

Driving Continuous Improvement

Driving for Results

Driving Projects to Completion

Interacting with People at Different Levels

Using Computers and Technology
"""

# print(llm.invoke("How can i create an application for resume checking for any JOB descrition"))
# checker = resume_checker(resume=resume, job_description=Job_description)
# print(checker)

st.title("Resume Screening Project")

# File uploader
# uploaded_file = st.file_uploader("Upload a PDF file ", type="pdf")
# # print(uploaded_file)
# st.write(load_pdf(uploaded_file))

pdf_path = st.text_input("Enter the path of the Resume")
resume = (load_pdf(pdf_path))
st.write(resume)

# Create a large text area for user input
Job_description = st.text_area("Enter Job Description:", height=300)
st.write(Job_description)

checker = resume_checker(resume=resume, job_description=Job_description)

st.write(checker)