from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
import spacy
import pathlib
import logging
import fitz
from docx import Document
from typing import List
import datetime
import platform

print("Starting FastAPI server...")

# Optional: Check the operating system and adjust paths accordingly
if platform.system() == "Windows":
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust this for your frontend origin if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Models...")

# Load existing models
# with open("model1.pkl", "rb") as file:
#     model1_data = pickle.load(file)
#     model1_tokenizer = model1_data["tokenizer"]
#     model1_model = model1_data["model"]
# print("Model 1 Loaded")

# with open("model3.pkl", "rb") as file:
#     models3 = pickle.load(file)
#     model3_tokenizer = models3["tokenizer"]
#     model3_model = models3["model"]
# print("Model 3 Loaded")

# Load Pegasus summarization model
pegasus_tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
pegasus_model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")
pegasus_pipeline = pipeline("summarization", model=pegasus_model, tokenizer=pegasus_tokenizer)
print("Pegasus Model Loaded")

# Load T5 summarization model
t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
t5_pipeline = pipeline("summarization", model=t5_model, tokenizer=t5_tokenizer)
print("T5 Model Loaded")

# Load Q&A model
qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)
print("Q&A Model Loaded")

nlp = spacy.load("en_core_web_sm")

# Utility functions
def extract_text_from_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(file_bytes):
    doc = Document(file_bytes)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# def summarize_with_model_1(text):
#     inputs = model1_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
#     outputs = model1_model.generate(inputs.input_ids, max_length=150, min_length=50, num_beams=4)
#     summary = model1_tokenizer.decode(outputs[0], skip_special_tokens=True)
#     key_points = extract_key_points(text)
#     return summary, key_points

# def summarize_with_model_3(text):
#     inputs = model3_tokenizer(f"Summarize: {text}", return_tensors="pt", truncation=True)
#     outputs = model3_model.generate(inputs.input_ids, max_length=150, num_beams=4)
#     summary = model3_tokenizer.decode(outputs[0], skip_special_tokens=True)
#     key_points = extract_key_points(text)
#     return summary, key_points

def summarize_with_pegasus(text):
    result = pegasus_pipeline(text, max_length=150, min_length=50, length_penalty=2.0, truncation=True)
    return result[0]["summary_text"]

def summarize_with_t5(text):
    result = t5_pipeline(text, max_length=150, min_length=50, length_penalty=2.0, truncation=True)
    return result[0]["summary_text"]

def extract_key_points(text):
    doc = nlp(text)
    key_points = []
    for sent in doc.sents:
        if any(ent.label_ in ["ORG", "PERSON", "GPE"] for ent in sent.ents):
            key_points.append(sent.text)
    return key_points

def qa_with_model(context, question):
    result = qa_pipeline(question=question, context=context)
    return result["answer"]

# Request Models
class SummarizeRequest(BaseModel):
    text: str
    model_type: str

class QARequest(BaseModel):
    context: str
    model_type: str
    question: str

print("All functions initialized")

@app.get("/")
async def read_main():
    logging.info("Endpoint '/' was called")
    return {"msg": "Hello World"}

# Endpoints
@app.post("/summarize")
async def summarize(file: UploadFile = File(...), model_type: str = Form(...)):
    content = await file.read()

    if file.content_type == "application/pdf":
        text = extract_text_from_pdf(content)
    elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx(content)
    else:
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="Unable to decode file content to text.")

    # if model_type == "model1":
        # summary, key_points = summarize_with_model_1(text)
    # elif model_type == "model3":
        # summary, key_points = summarize_with_model_3(text)
    if model_type == "pegasus":
        summary = summarize_with_pegasus(text)
        # key_points = extract_key_points(text)
    elif model_type == "t5":
        summary = summarize_with_t5(text)
        key_points = extract_key_points(text)
    else:
        raise HTTPException(status_code=400, detail="Invalid model type")
    # return {"summary": summary, "key_points": key_points}
    return {"summary": summary}

@app.post("/qa")
async def qa(request: QARequest):
    context = request.context
    question = request.question
    model_type = request.model_type

    if model_type == "model1" or model_type == "model3" or model_type == "pegasus" or model_type == "t5":
        answer = qa_with_model(context, question)
    else:
        raise HTTPException(status_code=400, detail="Invalid model type for Q&A")

    return {"answer": answer}

# History Management
analysis_history = []
class AnalysisReport(BaseModel):
    date: datetime.datetime
    model: str
    input_type: str
    input_text: str = None
    file_name: str = None
    summary: str
    key_points: List[str]

@app.get("/history")
async def get_history():
    if not analysis_history:
        raise HTTPException(status_code=404, detail="No analysis history found")
    return {"reports": analysis_history}

@app.delete("/clear-history")
async def clear_history():
    analysis_history.clear()
    return {"msg": "History cleared successfully"}

@app.post("/save-report")
async def save_report(report: AnalysisReport):
    analysis_history.append(report.dict())
    return {"msg": "Report saved successfully"}

print("All endpoints are ready!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
