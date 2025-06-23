import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

chain_gpt_35 = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1024)
chain_gpt_4_vision = ChatOpenAI(model="gpt-4o", max_tokens=1024)

from typing import Any
import os
from unstructured.partition.pdf import partition_pdf
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

input_path = os.getcwd()
output_path = os.path.join(os.getcwd(), "figures")

# Enhanced PDF processing with better table detection
print("Processing PDF with enhanced table detection...")

# Try different strategies for table detection
raw_pdf_elements = partition_pdf(
    filename=os.path.join(input_path, "startupai-financial-report-v2.pdf"),
    extract_images_in_pdf=True,
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=output_path,
    # Enhanced table detection parameters
    strategy="hi_res",  # High resolution processing
    hi_res_model_name="yolox",  # Better table detection model
)

import base64

text_elements = []
table_elements = []
image_elements = []

# Function to encode images
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Enhanced element classification with debugging
print("\nClassifying document elements...")
for i, element in enumerate(raw_pdf_elements):
    element_type = str(type(element))
    print(f"Element {i}: {element_type}")
    print(f"Content preview: {str(element)[:100]}...")
    print("---")
    
    # More comprehensive classification
    if 'CompositeElement' in element_type:
        text_elements.append(element)
    elif 'Table' in element_type:
        table_elements.append(element)
    elif 'FigureCaption' in element_type:
        text_elements.append(element)  # Treat captions as text
    else:
        # Check if content looks like a table based on structure
        content = str(element)
        if any(keyword in content.lower() for keyword in ['gross income', 'total expenses', 'net income', 'taxes']):
            print(f"Found potential table content in {element_type}")
            table_elements.append(element)
        else:
            text_elements.append(element)

# Extract text content
table_elements = [i.text if hasattr(i, 'text') else str(i) for i in table_elements]
text_elements = [i.text if hasattr(i, 'text') else str(i) for i in text_elements]

# Manual table extraction as fallback
def extract_financial_data_manually(text_content):
    """
    Extract financial table data manually if automatic detection fails
    """
    financial_tables = []
    
    # Look for financial data patterns
    for text in text_content:
        if any(keyword in text.lower() for keyword in ['gross income', 'total expenses', 'net income']):
            # This looks like financial data - treat as table
            financial_tables.append(text)
    
    return financial_tables

# If no tables detected, try manual extraction
if len(table_elements) == 0:
    print("No tables detected automatically. Attempting manual extraction...")
    manual_tables = extract_financial_data_manually(text_elements)
    if manual_tables:
        table_elements.extend(manual_tables)
        print(f"Manually extracted {len(manual_tables)} potential table(s)")

# Results
print(f"\nFinal count:")
print(f"Tables: {len(table_elements)}")
print(f"Text: {len(text_elements)}")

# Show table content if found
if table_elements:
    print("\nTable content found:")
    for i, table in enumerate(table_elements):
        print(f"Table {i+1}: {table[:200]}...")

# Process images
for image_file in os.listdir(output_path):
    if image_file.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(output_path, image_file)
        encoded_image = encode_image(image_path)
        image_elements.append(encoded_image)

print(f"Images: {len(image_elements)}")

from langchain.schema.messages import HumanMessage, AIMessage

# Function for text summaries
def summarize_text(text_element):
    prompt = f"Summarize the following text:\n\n{text_element}\n\nSummary:"
    response = chain_gpt_35.invoke([HumanMessage(content=prompt)])
    return response.content

# Enhanced table summarization
def summarize_table(table_element):
    prompt = f"""Analyze and summarize the following financial table/data:

{table_element}

Provide a clear summary that includes:
1. What type of financial information this contains
2. Key figures and amounts
3. Any important financial metrics or ratios

Summary:"""
    response = chain_gpt_35.invoke([HumanMessage(content=prompt)])
    return response.content

# Function for image summaries
def summarize_image(encoded_image):
    prompt = [
        AIMessage(content="You are a bot that is good at analyzing images."),
        HumanMessage(content=[
            {
                "type": "text", 
                "text": "Describe the contents of this image. If it contains financial data, tables, or charts, provide specific details about the numbers and structure."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                },
            },
        ])
    ]
    response = chain_gpt_4_vision.invoke(prompt)
    return response.content

# Processing with improved feedback
text_summaries = []
if text_elements:
    print(f"\nProcessing {len(text_elements)} text elements...")
    for i, te in enumerate(text_elements[:5]):  # Process more elements
        summary = summarize_text(te)
        text_summaries.append(summary)
        print(f"Text element {i + 1} processed.")
    
table_summaries = []
if table_elements:
    print(f"\nProcessing {len(table_elements)} table elements...")
    for i, te in enumerate(table_elements):
        summary = summarize_table(te)
        table_summaries.append(summary)
        print(f"Table element {i + 1} processed.")
        print(f"Table content: {te[:100]}...")
else:
    print("No table elements found to process.")
    
image_summaries = []
if image_elements:
    print(f"\nProcessing {len(image_elements)} image elements...")
    for i, ie in enumerate(image_elements[:8]):
        summary = summarize_image(ie)
        image_summaries.append(summary)
        print(f"Image element {i + 1} processed.")

import uuid
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma

# Initialize the Chroma vector database and docstore
vectorstorev2 = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())
storev2 = InMemoryStore()
id_key = "doc_id"

# Initialize the multi-vector retriever
retrieverv2 = MultiVectorRetriever(vectorstore=vectorstorev2, docstore=storev2, id_key=id_key)

# Function to add documents to the multi-vector retriever
def add_documents_to_retriever(summaries, original_contents, content_type):
    if not summaries:
        print(f"No {content_type} summaries to add - skipping.")
        return
    
    doc_ids = [str(uuid.uuid4()) for _ in summaries]
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i], "content_type": content_type})
        for i, s in enumerate(summaries)
    ]
    retrieverv2.vectorstore.add_documents(summary_docs)
    retrieverv2.docstore.mset(list(zip(doc_ids, original_contents)))
    print(f"{len(summaries)} {content_type} summaries added to retriever.")
    
# Add all content types
add_documents_to_retriever(text_summaries, text_elements[:len(text_summaries)], "text")
add_documents_to_retriever(table_summaries, table_elements, "table")
add_documents_to_retriever(image_summaries, image_summaries, "image")

# Test the enhanced system
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

template = """Answer the question based only on the following context, which can include text, images and tables:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

chain = (
    {"context": retrieverv2, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Enhanced question set
questions = [
    "What financial data is available in the tables?",
    "What is the company's gross income?",
    "What are the total expenses?",
    "What is the net income?",
    "How much did the company pay in taxes?",
    "What is the ROI percentage?",
    "What product does the company sell?",
    "Show me all the financial metrics from the tables."
]

print("\n" + "="*50)
print("TESTING ENHANCED MULTIMODAL RAG SYSTEM")
print("="*50)

for question in questions:
    print(f"\nQ: {question}")
    print("-" * 40)
    
    try:
        answer = chain.invoke(question)
        print(f"A: {answer}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("-" * 40)