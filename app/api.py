from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
from pathlib import Path
from ratelimit import limits, sleep_and_retry
import requests
import json
import logging
import sys
import os
import nltk
nltk.download('punkt')
import re

from nltk.tokenize import word_tokenize

books = [
    {
        "id": "1",
        "title": "Seven Effective habbits",
        "author": "Bill Will"
    },
    {
        "id": "2",
        "title": "Win Friends and Influence",
        "author": "Dale Carnie"
    }
]

class Authors(BaseModel):
    name: str
    
class SummaryRequest(BaseModel):
    title: str
    authors: list

TEN_MINUTES = 600 ###10 minutes

app = FastAPI(title='api')

load_dotenv(dotenv_path=Path(os.path.abspath(os.path.join(os.path.dirname(__file__),"../.env"))))

profanity_filter = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# StreamHandler for the console
stream_handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter("%(asctime)s [%(processName)s: %(process)d] [%(threadName)s: %(thread)d] [%(levelname)s] %(name)s: %(message)s")
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

logger.info('API is starting up')

origins = [
    "http://localhost:5173",
    "localhost:5173",
    "https://react-bookbot-ai.onrender.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

def preproces_query(query):
    """This is a method used for pre processing the text as tokens"""
    tokens = word_tokenize(query)
    return tokens

def keyword_extractor(text):
    """This method is used for extracting the keywords for building the query for api"""
    response = openai.chat.completions.create(
        model = "gpt-4",
        messages = [
            {
                "role": "user", "content": f"Extract keywords from the following text:\\n\\n{text}"
            }
        ]
    )
    keywords = response.choices[0].message.content.strip()
    logger.info(f"Extracted keywords {keywords}")
    return keywords

def get_book_summary(book_name, authors):
    """This method is used to provide a short summary for the book by authors"""
    
    if len(authors) == 0: 
        author = ''
    else:
        author = ', '.join(authors) if len(authors) > 1 else authors[0]
    
    logger.info("Generating summary for the book .. ")
    logger.info(f"author: {author}")
    logger.info(f"book title:{book_name}")
    
    prompt = f"Please provide a brief summary for the book with title {book_name} written by {author}"
    
    response = openai.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = [
            {
                "role": "user", "content": prompt
            }
        ]
    )
    summary = response.choices[0].message.content.strip()
    summary_es = re.sub('\\W+', '', summary.strip())
    logger.info(f"Books Summary {summary}")
    return summary;

#get_book_summary("How to Win Friends and Influence People", ["Dale Carnigie"])

def fetch_books_data(url):
    """Retreives books data from the openLibrary web API endpoint and returns the JSON response."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON response.")
        return None

def build_query(text):
    """This method prepares the query for Open Library api"""

    logger.info("Extracting words from text")
    tokens= preproces_query(text)
    
    logger.info("Retrieving the key words")
    extracted_keywords = keyword_extractor(text)
    
    query= re.sub('\\W+', '+', extracted_keywords.strip())
    logger.info(f"Generated query parameter {query}")
    return query

def build_api_endpoint(query, limit):
    """This is an method that constructs the api end point getting the response"""
    logger.info(f"Searching books for given input description : {query}")
    
    tokenized_query=build_query(query)
    
    api_url = f"https://openlibrary.org/search.json?q={tokenized_query}&fields=key,title,author_name,cover_i,cover_edition_key,first_publish_year&limit={limit}"
    
    return api_url

def check_profanity(query):
    """This is a method that validates the input text against the profanity and moderated the inputs"""
    response = profanity_filter.moderations.create(
        model="omni-moderation-latest",
        input=query
    )
    
    jsonFilter = json.loads(response.json())
    isLanguageAppropriate = jsonFilter["results"][0]["flagged"]
    if(isLanguageAppropriate):
        logger.info(f"We're sorry the language is in Appropriate")
    
    return isLanguageAppropriate
    
    
def search_books_by_query(query, limit):
    """Gets the books by searching the information for books"""
    
    logger.info(f"cheking the profanity ... ")
    isLanguageAppropriate = check_profanity(query)
    
    if(isLanguageAppropriate):
        return "We're sorry, but your input has been flagged as inappropriate. Please use a different phrase and try your request"

    data = fetch_books_data(build_api_endpoint(query, limit))
    books = data["docs"]
    
    logger.info(f"Size {len(books)}")
    
    for book in books:
        book["id"] = books.index(book)
        authors=[]
        if("author_name" in book):
            authors = book["author_name"]
        if("title" in book):
            book["summary"] = get_book_summary(book["title"], authors)
        else:
            book["summary"] = "Summary not available"
         
        if("cover_i" in book):
            book["cover_image_url"]="https://covers.openlibrary.org/b/id/"+str(book["cover_i"])+"-M.jpg"
        else:
            book["cover_image_url"]="https://covers.openlibrary.org/b/id/nocover-M.jpg"
    
    return books

@app.get("/", tags=["root"])
async def read_root() -> dict:
    logger.info('GET /')
    return {"message": "Welcome to your Books list."}

@app.get("/books", tags=["books"])
async def get_books() -> dict:
    logger.info("Getting books ...")
    return { "data": books }

@app.post("/books", tags=["books"])
async def add_book(book: dict) -> dict:
    books.append(book)
    return { 
        "books": { "Book added." } 
    }

@app.post("/summarize", tags=["summary"])
async def summarize_books(request: Request):
    logger.info("Summarizing books ..")
    req = await request.json()
    logger.info(req["title"])
    logger.info(req["authors"])
    return {
       "summary":get_book_summary(req["title"], req["authors"] )
    }    

@sleep_and_retry
@limits(calls=15, period=TEN_MINUTES)
@app.post("/search-books", tags=["searchBooks"])
async def search_books(request: Request):
    logger.info("Received a request for books search")
    req= await request.json()
    text= req["query"]
    resultCount= req["books_limit"]
    logger.info("Search for books")
    
    fetched = search_books_by_query(text, resultCount)
        
    return {
        "books": fetched
    }