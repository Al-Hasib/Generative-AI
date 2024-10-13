import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
import fitz
#pip install pymupdf
import streamlit as st
import pandas as pd
import json



prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You will compare the Resume and Job Description and provide the Output in a JSON format"),
    ("human", "You will take the resume and Job description: \n \
     Resume: {resume} \n \
     Job Description: {Job_description} \n \
     Based on the Job description, does the resume is perfect for the JOB and What's the percentage of Matching the resume against the JOB description. \
     The JSON Output Key will be Name of resume holder, email, is_perfect,is_okay, Matching Score in percentage, strong zone, Lack of Knowledge")
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
def resume_checker(resume, job_description,llm):
    Chain = prompt_template | llm | parser
    result = Chain.invoke({"resume":resume, "Job_description":job_description})
    return result



def main():
    load_dotenv()
    api_key = os.getenv("GOOGLE_GEN_API")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=api_key)
    # print(llm.invoke("How can i create an application for resume checking for any JOB descrition"))
    # checker = resume_checker(resume=resume, job_description=Job_description)
    # print(checker)

    st.title("Resume Screening Project")

    # File uploader
    # uploaded_file = st.file_uploader("Upload a PDF file ", type="pdf")
    # # print(uploaded_file)
    # st.write(load_pdf(uploaded_file))

    folder_path = st.text_input("Enter the path of the Resume folder")
    Job_description = st.text_area("Enter Job Description:", height=300)
    information_list = []
    for i in os.listdir(folder_path):
        pdf_path = os.path.join(folder_path,i)
        resume = (load_pdf(pdf_path))
        # st.write(resume)

        # Create a large text area for user input
        # st.write(Job_description)

        checker = resume_checker(resume=resume, job_description=Job_description,llm=llm)

        print(type(checker))
        st.write(checker)

        checker = checker.replace('```json', "").replace('```', "")

        # Convert to DataFrame
        data = json.loads(str(checker))
        information_list.append(data)

    # df = pd.DataFrame({
    #     "Name of resume holder": [data["Name of resume holder"]],
    #     "email": [data["email"]],
    #     "is_perfect": [data["is_perfect"]],
    #     "is_okay": [data["is_okay"]],
    #     "Matching Score in percentage": [data["Matching Score in percentage"]],
    #     "strong zone": ["\n".join(data["strong zone"])],
    #     "Lack of Knowledge": ["\n".join(data["Lack of Knowledge"])]
    # })

    df = pd.DataFrame(information_list)

    df.to_csv("resume screening.csv", index=False)

    # Display the dataframe
    st.write(df)




if __name__=="__main__":
    main()