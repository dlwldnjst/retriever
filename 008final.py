import pandas as pd
import streamlit as st
import torch
import logging
import re
import sqlite3
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from cachetools import cached, TTLCache
import google.generativeai as genai
import requests
from tornado.websocket import WebSocketClosedError
import atexit
import os
import tempfile
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import traceback 

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def clear_memory():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()  # MPS에서 캐시를 비웁니다.
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()  # CUDA에서 캐시를 비웁니다.

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Streamlit secrets에서 API 키 불러오기
gemini_api_key = st.secrets["GEMINI_API_KEY"]

if not gemini_api_key:
    logger.error("GEMINI_API_KEY is not set")
    st.error("Gemini API key is not configured. Please set the GEMINI_API_KEY.")

# Gemini API 설정
genai.configure(api_key=gemini_api_key)

# Kakao API 설정
kakao_api_url = st.secrets["kakao"]["API_URL"]
kakao_rest_api_key = st.secrets["kakao"]["REST_API_KEY"]
HEADERS = {"Authorization": f"KakaoAK {kakao_rest_api_key}"}

# device 변수 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Streamlit 페이지 설정
st.set_page_config(layout="wide")

# 세션 상태 초기화
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'user_books' not in st.session_state:
    st.session_state.user_books = pd.DataFrame()
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = ""  # 초기화 필요
if 'keywords' not in st.session_state:
    st.session_state.keywords = []  # 초기화 필요
if 'filtered_books' not in st.session_state:
    st.session_state.filtered_books = pd.DataFrame()  # 초기화 필요
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = ""  # 초기화 필요
if 'books' not in st.session_state:
    st.session_state.books = []  # 초기화 필요
if 'books_info' not in st.session_state:
    st.session_state.books_info = []  # 초기화 필요
if 'show_tables' not in st.session_state:
    st.session_state.show_tables = False  # 초기화 필요
if 'saved_recommendations' not in st.session_state:
    st.session_state.saved_recommendations = ""  # 초기화 필요
if 'response' not in st.session_state:
    st.session_state.response = ""  # 초기화 필요
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""  # 초기화 필요
if 'filtered_books_in_collection' not in st.session_state:
    st.session_state.filtered_books_in_collection = pd.DataFrame()
if 'db_connection' not in st.session_state:
    st.session_state.db_connection = None  # 초기화 필요
if 'temp_db_path' not in st.session_state:
    st.session_state.temp_db_path = ""  # 초기화 필요

# 초기화 상태 확인을 위한 디버그 로그
logger.debug(f"Session state after initialization: {st.session_state}")

def connect_to_uploaded_db(uploaded_file):
    if uploaded_file is not None:
        try:
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # st.write(f"임시 파일 경로: {tmp_file_path}")
            
            # SQLite 연결
            conn = sqlite3.connect(tmp_file_path)
            
            # 연결 테스트
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            # st.write(f"데이터베이스 테이블: {tables}")
            
            # 세션 상태에 데이터베이스 연결과 경로 저장
            st.session_state.db_connection = conn
            st.session_state.temp_db_path = tmp_file_path
            
            return conn, tmp_file_path
        except Exception as e:
            st.error(f"데이터베이스 연결 중 오류 발생: {str(e)}")
            logger.error(f"Database connection error: {str(e)}")
            return None, None
    return None, None

# 데이터베이스 초기화 함수
def init_db(conn):
    if conn is None:
        logger.error("데이터베이스 연결이 없습니다.")
        return
    
    c = conn.cursor()
    try:
        c.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS books_fts USING fts5(
                title, 
                author, 
                description, 
                isbn
            );
        ''')
        conn.commit()
        logger.info("books_fts 테이블이 성공적으로 생성되었습니다.")
        st.success("books_fts 테이블이 성공적으로 생성되었습니다.")
    except sqlite3.Error as e:
        logger.error(f"books_fts 테이블 생성 중 오류 발생: {e}")
        st.error(f"books_fts 테이블 생성 중 오류 발생: {e}")

# 데이터 로드 및 FTS 테이블 업데이트 함수
def load_and_update_fts(conn):
    if conn:
        c = conn.cursor()

        try:
            c.execute("SELECT title, author, description, isbn FROM books")
            books_data = c.fetchall()
            
            logger.info(f"Loaded {len(books_data)} books from the database")

            c.execute("DELETE FROM books_fts")

            c.executemany('''
                INSERT INTO books_fts (title, author, description, isbn)
                VALUES (?, ?, ?, ?)
            ''', books_data)

            conn.commit()
            logger.info(f"Successfully updated books_fts with {len(books_data)} records")
            st.success(f"books_fts 테이블이 {len(books_data)}개의 레코드로 업데이트되었습니다.")
        except sqlite3.Error as e:
            logger.error(f"FTS 테이블 업데이트 중 오류 발생: {e}")
            st.error(f"FTS 테이블 업데이트 중 오류 발생: {e}")

def cleanup_temp_file():
    if 'db_connection' in st.session_state:
        st.session_state.db_connection.close()
    if 'temp_db_path' in st.session_state:
        os.unlink(st.session_state.temp_db_path)

atexit.register(cleanup_temp_file)

def cleanup():
    if 'db_connection' in st.session_state and st.session_state.db_connection is not None:
        st.success("데이터베이스 연결이 성공적으로 설정되었습니다.")
    else:
        st.error("데이터베이스 연결이 설정되지 않았습니다.")
    if 'db_connection' in st.session_state and st.session_state.db_connection:
        st.session_state.db_connection.close()
    if 'temp_db_path' in st.session_state:
        os.unlink(st.session_state.temp_db_path)

atexit.register(cleanup)

# 유저 입력을 전처리하는 함수
def preprocess_input(text):
    return re.sub(r'\s+', ' ', text).strip()

# API 호출 결과 캐싱
cache = TTLCache(maxsize=100, ttl=300)

def extract_keywords(user_input):
    prompt = (
        f"다음 문장에서 도서 검색을 위한 키워드를 추출해: \"{user_input}\". "
        "다음 단어는 키워드에서 제외시켜 : '책' '추천' '대한'"
        "키워드만 반환해. 콤마로 구분"
    )
    response = call_gemini_api_cached(prompt)
    if response:
        keywords = [keyword.strip() for keyword in response.split(',') if keyword.strip()]
        logger.info(f"추출된 키워드: {keywords}")
        return keywords
    else:
        logger.warning("키워드 추출 실패")
        return []

def search_books(keywords):
    logger.info(f"Searching for keywords: {keywords}")

    if 'db_connection' not in st.session_state or st.session_state.db_connection is None:
        logger.error("데이터베이스 연결이 없습니다.")
        st.session_state.error_message = "데이터베이스 연결이 없습니다. 파일을 다시 업로드해주세요."
        return pd.DataFrame()

    conn = st.session_state.db_connection
    c = conn.cursor()

    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='books_fts';")
    if not c.fetchone():
        logger.error("books_fts 테이블이 존재하지 않습니다.")
        st.session_state.error_message = "books_fts 테이블이 존재하지 않습니다. 데이터베이스 구조를 확인해주세요."
        return pd.DataFrame()

    if not keywords:
        logger.warning("검색할 키워드가 없습니다.")
        return pd.DataFrame()

    combined_keywords = ' AND '.join(keywords)

    full_query = f'''
        SELECT title, author, description, isbn
        FROM books_fts 
        WHERE books_fts MATCH ?;
    '''

    params = [combined_keywords]

    logger.info(f"실행할 쿼리: {full_query}")
    logger.info(f"쿼리 파라미터: {params}")

    try:
        c.execute(full_query, params)
        results = c.fetchall()
        logger.info(f"검색 결과 수: {len(results)}")

        if not results:
            st.session_state.warning_message = "검색 결과가 없습니다."
            logger.warning("검색 결과가 없습니다.")

        df = pd.DataFrame(results, columns=['title', 'author', 'description', 'isbn'])
        logger.info(f"검색 결과 DataFrame 크기: {df.shape}")
        return df
    except sqlite3.OperationalError as e:
        logger.error(f"SQLite 오류: {e}")
        st.session_state.error_message = f"데이터베이스 검색 중 오류 발생: {str(e)}"
        return pd.DataFrame()

@st.cache_data

def filter_and_add_call_numbers(search_results, user_books):
    logger.debug(f"search_results: {search_results}")
    logger.debug(f"user_books: {user_books}")
    
    if search_results is None or search_results.empty:
        logger.error("검색 결과가 비어 있습니다.")
        return pd.DataFrame()
    
    if user_books.empty:
        logger.error("소장 도서 목록이 비어 있습니다.")
        return pd.DataFrame()

    isbn_col = next((col for col in user_books.columns if 'isbn' in col.lower()), None)
    call_number_col = next((col for col in user_books.columns if '청구기호' in col), None)

    if not isbn_col or not call_number_col:
        logger.error("ISBN 또는 청구기호 컬럼을 찾을 수 없습니다.")
        return pd.DataFrame()

    search_results['processed_isbn'] = search_results['isbn'].astype(str).str.replace('-', '').str.lower()
    user_books['processed_isbn'] = user_books[isbn_col].astype(str).str.replace('-', '').str.lower()

    merged = pd.merge(search_results, user_books[['processed_isbn', call_number_col]], on='processed_isbn', how='left')

    filtered_books = merged.dropna(subset=[call_number_col])
    filtered_books.rename(columns={call_number_col: '청구기호'}, inplace=True)

    logger.info(f"총 도서 수: {len(merged)}, 청구기호가 있는 도서 수: {len(filtered_books)}")
    
    return filtered_books if not filtered_books.empty else pd.DataFrame()

def get_book_recommendations(user_input, filtered_books):
    if filtered_books is None or filtered_books.empty:
        return "죄송합니다. 요청하신 조건에 맞는 도서를 찾지 못했습니다."

    book_list = "\n".join([
        f"ISBN: {book['isbn']}, 제목: {book['title']}, 저자: {book['author']}, 청구기호: {book['청구기호']}, 설명: {book['description']}"
        for _, book in filtered_books.iterrows()
    ])

    prompt = f"""
    다음 도서 목록을 바탕으로 사용자의 요청에 맞는 책을 추천해주세요.
    지침:
    1. 각 책에 대해 다음 내용을 포함하여 최소 5문장 이상의 상세한 설명을 제공해 주세요:
       - 책의 주요 내용 요약
       - 책이 다루는 핵심 주제나 사건
       - 책의 구성이나 특징적인 부분
       - 이 책이 독자에게 줄 수 있는 통찰이나 가치
       - 추천 이유
    2. 최대 5권의 책을 추천해주세요.
    3. 한글로만 답해주세요.
    4. 이용자가 업로드한 도서 목록에 포함된 책만 추천해주세요. 목록에 없는 책은 절대로 추천하지 마세요.
    5. 책 제목이나 저자를 임의로 바꾸지 말고 그대로 가지고 오세요.
    6. 시리즈로 된 책은 1권만 추천하세요.
    7. 각 추천은 다음 형식으로 정확히 작성해주세요:
        [BOOK_START]
        ISBN: [ISBN]
        제목: [제목]
        저자: [저자]
        청구기호: [청구기호]
        설명: [설명]
        [BOOK_END]

    사용자가 업로드한 도서 목록:
    {book_list}

    사용자 요청: {user_input}

    각 책 추천은 반드시 [BOOK_START]로 시작하고 [BOOK_END]로 끝나야 합니다.
    모든 책 정보는 위의 형식을 정확히 따라야 하며, 어떤 추가 포맷팅(예: 볼드체)도 사용하지 마세요."""

    response = call_gemini_api_cached(prompt)
    return response


def extract_book_info_from_recommendations(recommendations, filtered_books):
    # ISBN 추출
    isbn_pattern = r'\b(?:ISBN(?:-1[03])?:? )?(?=[0-9X]{10}$|(?=(?:[0-9]+[- ]){3})[- 0-9X]{13}$|97[89][0-9]{10}$|(?=(?:[0-9]+[- ]){4})[- 0-9]{17}$)(?:97[89][- ]?)?[0-9]{1,5}[- ]?[0-9]+[- ]?[0-9]+[- ]?[0-9X]\b'
    isbns = re.findall(isbn_pattern, recommendations)
    
    if not isbns:
        logger.warning("No ISBNs found in recommendations")
    
    book_info_list = []
    for isbn in isbns:
        # ISBN을 기반으로 filtered_books에서 책 정보 찾기
        book = filtered_books[filtered_books['isbn'] == isbn]
        
        if not book.empty:
            # 책 정보가 있는 경우
            info = {
                'title': book['title'].iloc[0],
                'author': book['author'].iloc[0],
                'isbn': isbn,
                'call_number': book['call_number'].iloc[0] if 'call_number' in book.columns else 'N/A'
            }
        else:
            # 책 정보가 없는 경우, Gemini 응답에서 제목과 저자 추출 시도
            title_match = re.search(fr'(?:제목|책 제목):\s*(.*?)(?:\n|$|{isbn})', recommendations)
            author_match = re.search(fr'(?:저자|작가):\s*(.*?)(?:\n|$|{isbn})', recommendations)
            
            info = {
                'title': title_match.group(1) if title_match else 'Unknown Title',
                'author': author_match.group(1) if author_match else 'Unknown Author',
                'isbn': isbn,
                'call_number': 'N/A'
            }
        
        book_info_list.append(info)
    
    if not book_info_list:
        logger.warning("No book information could be extracted from recommendations")
    
    # st.write(f"추출된 책 정보: {book_info_list}")  # 디버깅용
    
    return book_info_list


def save_recommendations(recommendations):
    # Gemini의 응답을 세션 상태에 저장
    st.session_state.saved_recommendations = recommendations

def unload_model(model):
    for param in model.parameters():
        param.grad = None
    model.to('cpu')
    del model
    torch.cuda.empty_cache()  # 또는 torch.mps.empty_cache() 

def parse_saved_recommendations():
    if 'saved_recommendations' not in st.session_state:
        return []

    recommendations = st.session_state.saved_recommendations
    
    # ISBN 추출
    isbn_pattern = r'\b(?:ISBN(?:-1[03])?:? )?(?=[0-9X]{10}$|(?=(?:[0-9]+[- ]){3})[- 0-9X]{13}$|97[89][0-9]{10}$|(?=(?:[0-9]+[- ]){4})[- 0-9]{17}$)(?:97[89][- ]?)?[0-9]{1,5}[- ]?[0-9]+[- ]?[0-9]+[- ]?[0-9X]\b'
    isbns = re.findall(isbn_pattern, recommendations)
    
    book_info_list = []
    for isbn in isbns:
        title_match = re.search(fr'제목:\s*(.*?)(?:\n|$|{isbn})', recommendations)
        author_match = re.search(fr'저자:\s*(.*?)(?:\n|$|{isbn})', recommendations)
        
        info = {
            'title': title_match.group(1) if title_match else 'Unknown Title',
            'author': author_match.group(1) if author_match else 'Unknown Author',
            'isbn': isbn,
            'call_number': 'N/A'  # 청구기호는 별도로 처리해야 할 수 있습니다
        }
        book_info_list.append(info)
    
    return book_info_list

async def extract_keywords_async(user_input):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, extract_keywords, user_input)

async def search_books_async(keywords):
    loop = asyncio.get_event_loop()
    search_results = await loop.run_in_executor(None, search_books, keywords)
    logger.debug(f"search_results: {search_results}")
    if search_results is None or search_results.empty:
        logger.warning("검색 결과가 비어있습니다.")
        st.warning("검색 결과가 없습니다. 다른 키워드로 시도해보세요.")
    return search_results if search_results is not None else pd.DataFrame()

async def get_book_description_async(isbn):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, get_book_recommendations, isbn)

def process_user_input_sync(user_input):
    processed_input = preprocess_input(user_input)
    
    try:
        with st.spinner('리트리버가 책을 찾고 있어요...'):
            # Keyword extraction
            st.write("1. 키워드 추출 중...")
            extracted_keywords = extract_keywords(processed_input)
            st.session_state.keywords = extracted_keywords
            # st.write(f"추출된 키워드: {st.session_state.keywords}")
            
            if not st.session_state.keywords:
                st.warning("키워드를 추출하지 못했습니다. 다른 질문을 시도해보세요.")
                return

            # Book search
            st.write("2. 키워드로 도서 검색 중...")
            search_results = search_books(st.session_state.keywords)
            
            if search_results is None or search_results.empty:
                st.warning("검색 결과가 없습니다. 다른 키워드로 시도해보세요.")
                return
            
            # st.write(f"검색된 도서 수: {len(search_results)}")

            # Filtering and adding call numbers
            try:
                st.write("3. 소장 도서 판별 및 청구기호 추가 중...")
                filtered_books = filter_and_add_call_numbers(search_results, st.session_state.user_books)
                st.session_state.filtered_books = filtered_books
                
                if filtered_books is None or filtered_books.empty:
                    st.warning("우리 도서관에는 관련된 책이 없습니다.")
                    return
            except Exception as e:
                st.error(f"소장 도서 판별 중 오류가 발생했습니다: {str(e)}")
                logger.error(f"Exception during filtering books: {traceback.format_exc()}")
                return

            # Vector search
            try:
                st.write("4. 벡터 검색을 통한 상위 도서 추출 중...")
                top_books = search_with_vector(' '.join(st.session_state.keywords), top_k=10)
                st.session_state.filtered_books = top_books

                if top_books is None or top_books.empty:
                    st.warning("우리 도서관에는 관련된 책이 없습니다.")
                    return
            except Exception as e:
                st.error(f"벡터 검색 중 오류가 발생했습니다: {str(e)}")
                logger.error(f"Exception during vector search: {traceback.format_exc()}")
                return
            
            # Book recommendations
            try:
                st.write("5. 최종 도서 추천 중...")
                st.session_state.recommendations = get_book_recommendations(' '.join(st.session_state.keywords), st.session_state.filtered_books)
            except Exception as e:
                st.error(f"도서 추천 중 오류가 발생했습니다: {str(e)}")
                logger.error(f"Exception during book recommendations: {traceback.format_exc()}")
                return
    
    except Exception as e:
        st.error(f"처리 중 예상치 못한 오류가 발생했습니다: {str(e)}")
        logger.error(f"Exception: {traceback.format_exc()}")
        return

    st.success("책 추천이 완료되었습니다. 아래에서 결과를 확인하세요.")

    if not st.session_state.recommendations:
        st.error("추천된 도서가 없습니다.")
        return

    try:
        st.session_state.books = parse_response(st.session_state.recommendations)
        if not st.session_state.books:
            st.error("추천된 도서 정보를 파싱하는 중 오류가 발생했습니다.")
            return
    except Exception as e:
        st.error(f"추천된 도서 정보를 파싱하는 중 오류가 발생했습니다: {str(e)}")
        logger.error(f"Exception during parsing recommendations: {traceback.format_exc()}")
        return

    try:
        st.session_state.books_info = fetch_book_info_batch(st.session_state.books, HEADERS, kakao_api_url, st.session_state.user_books)
        if st.session_state.books_info is None:
            st.error("도서 정보 배치를 가져오는 중 오류가 발생했습니다.")
            return
    except Exception as e:
        st.error(f"도서 정보 배치를 가져오는 중 오류가 발생했습니다: {str(e)}")
        logger.error(f"Exception during fetching book info: {traceback.format_exc()}")
        return

    st.session_state.show_tables = True

    if not st.session_state.books_info.empty:
        st.subheader("추천 도서")
        display_books(st.session_state.books_info)
    else:
        st.warning("추천할 도서가 없습니다.")


def display_books(books):
    if books.empty:
        st.write("표시할 책 정보가 없습니다.")
        return

    for index, book in books.iterrows():
        col1, col2 = st.columns([1, 3])

        with col1:
            if "thumbnail" in book and book["thumbnail"]:
                st.image(book["thumbnail"], width=150)
            else:
                st.write("표지 이미지 없음")

        with col2:
            st.subheader(book["title"])
            st.write(f"저자: {book['author']}")
            st.write(f"ISBN: {book['isbn']}")
            st.write(f"청구기호: {book['call_number']}")
            st.write(f"{book['contents'][:500]}" if len(book['contents']) > 200 else book['contents'])

        st.markdown("---")


def parse_response(response):
    if not response:
        logger.error("응답이 없습니다.")
        return []

    books = []
    book_pattern = r'\[BOOK_START\](.*?)\[BOOK_END\]'
    book_matches = re.findall(book_pattern, response, re.DOTALL)
    
    for book_text in book_matches:
        book = {}
        patterns = {
            'isbn': r'ISBN:\s*(\S+)',
            'title': r'제목:\s*(.*?)\n',
            'author': r'저자:\s*(.*?)\n',
            'contents': r'설명:\s*(.*?)\n'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, book_text, re.DOTALL)
            if match:
                book[key] = match.group(1).strip()
        
        if book:
            books.append(book)
    
    logger.debug(f"Parsed books: {books}")
    return books

def fetch_book_info_batch(books, headers, api_url, user_books_df):
    if not books:
        logger.error("책 정보 리스트가 비어 있습니다.")
        return pd.DataFrame()

    book_info_list = []

    def fetch_info(book):
        isbn = book.get('isbn', '').replace(',', '').replace(' ', '').lower()
        if not isbn:
            return None, f"ISBN이 없는 책 정보가 있습니다: {book}"

        params = {"target": "isbn", "query": isbn}
        try:
            response = requests.get(api_url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                if data["documents"]:
                    api_book_info = data["documents"][0]
                    
                    matching_row = user_books_df[user_books_df['processed_isbn'].str.contains(isbn, na=False)]
                    if matching_row.empty:
                        matching_row = user_books_df[user_books_df['processed_isbn'].str.endswith(isbn[-9:], na=False)]
                    
                    call_number = matching_row['청구기호'].values[0] if not matching_row.empty else 'N/A'
                    
                    if call_number == 'N/A':
                        return None, f"ISBN {isbn}에 대한 청구기호를 찾지 못했습니다. 이 책은 추천 목록에서 제외됩니다."

                    return {
                        "title": book.get('title', api_book_info.get('title', '')),
                        "author": book.get('author', api_book_info.get('authors', [''])[0]),
                        "contents": book.get('contents', api_book_info.get('contents', '')),
                        "isbn": isbn,
                        "thumbnail": api_book_info.get("thumbnail", ""),
                        "call_number": call_number
                    }, None
                else:
                    return None, f"API에서 ISBN {isbn}에 대한 정보를 찾지 못했습니다. 이 책은 추천 목록에서 제외됩니다."
            else:
                return None, f"API 요청 실패 (상태 코드: {response.status_code}). 이 책은 추천 목록에서 제외됩니다."
        except Exception as e:
            return None, f"ISBN: {isbn}에 대한 요청 중 예외가 발생했습니다: {e}. 이 책은 추천 목록에서 제외됩니다."

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_book = {executor.submit(fetch_info, book): book for book in books}
        for future in as_completed(future_to_book):
            book_info, warning_message = future.result()
            if book_info:
                book_info_list.append(book_info)
            if warning_message:
                st.warning(warning_message)
            time.sleep(0.1)  # API rate limiting을 위한 딜레이

    # st.write(f"최종 책 정보 목록: {book_info_list}")  # 디버깅을 위해 최종 정보 출력
    return pd.DataFrame(book_info_list)

# 텍스트 임베딩 함수
def get_text_embedding(text):
    embedding = model.encode(text)
    return np.array(embedding).flatten()  # 1D 배열로 변환

# 벡터 검색 함수
def vector_search(query_embedding, embeddings, top_k=10):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    _, indices = index.search(query_embedding, top_k)
    return indices[0]

# 벡터 검색 통합 함수
def search_with_vector(query, top_k=10):
    try:
        keywords = query.split()
        search_results = search_books(keywords)
        
        if search_results.empty:
            st.write("검색 결과가 없습니다.")
            logger.error("검색 결과가 없습니다.")
            return pd.DataFrame()

        search_results = search_results.head(10)
        logger.debug(f"Search results after head: {search_results}")

        query_embedding = get_text_embedding(query)
        logger.debug(f"Query embedding: {query_embedding}")

        embeddings = [get_text_embedding(desc) for desc in search_results['description']]
        embeddings = np.array(embeddings)
        logger.debug(f"Generated embeddings: {embeddings}")

        if embeddings.size == 0:
            st.write("임베딩 생성 실패")
            logger.error("임베딩 생성 실패")
            return pd.DataFrame()

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        logger.debug(f"FAISS index created with dimension: {dim}")

        distances, indices = index.search(query_embedding.reshape(1, -1), min(top_k, len(embeddings)))
        logger.debug(f"Search indices: {indices}")
        logger.debug(f"Search distances: {distances}")

        top_results = search_results.iloc[indices[0]]
        logger.debug(f"Top results: {top_results}")

        filtered_books = filter_and_add_call_numbers(top_results, st.session_state.user_books)
        logger.debug(f"Filtered books: {filtered_books}")

        if filtered_books.empty:
            st.warning("우리 도서관에는 관련된 책이 없습니다.")
            logger.error("필터링된 도서 목록이 비어 있습니다.")
            return pd.DataFrame()

        return filtered_books

    except Exception as e:
        st.error(f"검색 중 오류가 발생했습니다: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        logger.error(f"Exception during search with vector: {e}")
        return pd.DataFrame()

def call_gemini_api(prompt, retries=3, wait_time=10):
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    if not prompt.strip():
        return "죄송합니다. 요청하신 조건에 맞는 도서를 찾지 못했습니다."

    logs = []
    result = None
    for attempt in range(retries):
        try:
            logs.append(f"Attempt {attempt + 1} to call Gemini API with prompt: {prompt}")
            response = model.generate_content(contents=prompt)
            if response:
                if hasattr(response, 'text') and response.text:
                    logs.append(f"Received text response: {response.text}")
                    result = response.text
                    break
                elif hasattr(response, 'parts'):
                    if response.parts:
                        part_text = response.parts[0].text if response.parts else "죄송합니다. 요청하신 조건에 맞는 도서를 찾지 못했습니다."
                        logs.append(f"Received parts response: {part_text}")
                        result = part_text
                        break
                    else:
                        logs.append("API response error: No parts in response")
                        if hasattr(response, 'prompt_feedback'):
                            logs.append(f"Prompt feedback: {response.prompt_feedback}")
                else:
                    if hasattr(response, 'prompt_feedback'):
                        logs.append(f"API response error: {response.prompt_feedback}")
                    else:
                        logs.append("API response error: No text or parts in response")
            else:
                logs.append("Received empty response")
        except Exception as e:
            if "Resource has been exhausted" in str(e) and attempt < retries - 1:
                logs.append("Resource exhausted, waiting before retrying...")
                time.sleep(wait_time)
            else:
                logs.append(f"Exception during API call: {e}")
                if hasattr(response, 'prompt_feedback'):
                    logs.append(f"Prompt feedback: {response.prompt_feedback}")
                break

    if result is None:
        result = "죄송합니다. 요청하신 조건에 맞는 도서를 찾지 못했습니다."

    return result

@cached(cache)
def call_gemini_api_cached(prompt):
    response = call_gemini_api(prompt)
    if not response:
        logger.error("Gemini API 응답이 없습니다.")
    return response


def main():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #fcf8f0;
            color: #333333;
        }
        .user-message, .bot-message {
            border-radius: 5px;
            padding: 10px;
            margin: 5px 0;
            text-align: left;
            background: none;
        }
        .chat-container {
            max-height: 300px;
            overflow-y: auto;
            padding: 10px;
            margin-right: 250px;
        }
        .input-container {
            width: 100%;
            background: #fcf8f0;
            padding: 10px;
            margin-right: 10px;
        }
        .input-container input {
            width: 100%;
        }
        .image-container {
            position: relative;
            width: 220px;
            height: auto;
            z-index: 1000;
            margin-left: auto;
        }
        .upload-container {
            width: 100%;
            padding: 10px;
        }
        .hidden-title {
            display: none;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: transparent;
            text-align: center;
            color: gray;
            font-size: 14px;
            cursor: pointer;
        }
        .hidden-email {
            display: none;
        }
        </style>
        <script>
        function toggleEmail() {
            var emailElement = document.getElementById('email');
            if (emailElement.style.display === 'none' || emailElement.style.display === '') {
                emailElement.style.display = 'inline';
            } else {
                emailElement.style.display = 'none';
            }
        }
        </script>
        """,
        unsafe_allow_html=True
    )

    st.title("리트리버: 책을 찾아줘 0.08 ver")

    # 컬럼 레이아웃 설정
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        user_books_file = st.file_uploader("소장 도서 목록 업로드", type=["xlsx"])
        db_file = st.file_uploader("도서 데이터베이스 업로드 (SQLite 파일)", type=["db", "sqlite", "sqlite3"])

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if user_books_file is not None and db_file is not None:
            try:
                # 엑셀 파일 처리
                st.session_state.user_books = pd.read_excel(user_books_file)
                st.success('소장 도서 목록을 성공적으로 업로드했습니다.')

                # SQLite 파일 처리
                st.session_state.db_connection, st.session_state.temp_db_path = connect_to_uploaded_db(db_file)
                if st.session_state.db_connection:
                    c = st.session_state.db_connection.cursor()
                    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = c.fetchall()
                    # st.write(f"데이터베이스 테이블: {tables}")
                    
                    if ('books_fts',) in tables:
                        c.execute("PRAGMA table_info(books_fts);")
                        columns = c.fetchall()
                        # st.write(f"books_fts 테이블 구조: {columns}")
                    else:
                        st.error("books_fts 테이블이 존재하지 않습니다.")
                    
                    # FTS 테이블 확인
                    cursor = st.session_state.db_connection.cursor()
                    cursor.execute("SELECT COUNT(*) FROM books_fts;")
                    count = cursor.fetchone()[0]
                    # st.write(f"FTS 테이블의 총 레코드 수: {count}")
                
                # ISBN 열 찾기 (대소문자 구분 없이)
                isbn_column = next((col for col in st.session_state.user_books.columns if col.lower() == 'isbn'), None)
                if isbn_column is None:
                    st.error("업로드된 파일에 'ISBN' 또는 'isbn' 컬럼이 없습니다. 파일을 확인해 주세요.")
                    return

                # 청구기호 열 찾기 (대소문자 구분 없이)
                call_number_column = next((col for col in st.session_state.user_books.columns if col == '청구기호'), None)
                if call_number_column is None:
                    st.error("업로드된 파일에 '청구기호' 컬럼이 없습니다. 파일을 확인해 주세요.")
                    return

                # ISBN 데이터 전처리
                st.session_state.user_books[isbn_column] = st.session_state.user_books[isbn_column].astype(str)
                st.session_state.user_books[isbn_column] = st.session_state.user_books[isbn_column].str.replace(',', '').str.replace(' ', '')
                st.session_state.user_books['processed_isbn'] = st.session_state.user_books[isbn_column].apply(lambda x: str(int(float(x))) if '.' in x else x).str.lower()

                # 청구기호 데이터 전처리
                st.session_state.user_books[call_number_column] = st.session_state.user_books[call_number_column].astype(str)

                st.success('데이터베이스에서 도서 정보를 성공적으로 로드했습니다.')

                st.markdown('<div class="input-container">', unsafe_allow_html=True)
                with st.form(key='chat_form', clear_on_submit=True):
                    user_input = st.text_input("질문을 입력해 주세요:", key="user_input_form")
                    submit_button = st.form_submit_button(label='검색')

                if submit_button and user_input:
                    try:
                        results = process_user_input_sync(user_input)

                        # 이전 검색 결과 초기화
                        st.session_state.filtered_books = pd.DataFrame()
                        st.session_state.recommendations = ""
                        

                    except WebSocketClosedError:
                        st.error("웹소켓 연결이 닫혔습니다. 다시 시도해 주세요.")
            except Exception as e:
                        st.error(f"처리 중 예상치 못한 오류가 발생했습니다: {str(e)}")
                        logger.exception("Unexpected error in main execution")

        else:
            st.info("소장 도서 목록(엑셀)과 도서 데이터베이스(SQLite)를 모두 업로드해 주세요.")

    with col2:
        st.markdown(
            f'<div class="image-container"><img src="https://blog.kakaocdn.net/dn/dDYRPT/btsH9gsvNeX/JN4pkLNXXgyjHW912WUyGk/img.png" style="width: 100%; height: auto;"></div>',
            unsafe_allow_html=True
        )

    st.components.v1.html(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: transparent;
            text-align: center;
            color: gray;
            font-size: 14px;
            cursor: pointer;
        }
        .hidden-email {
            display: none;
        }
        .footer span {
            margin-right: 20px;
        }
        </style>
        <div class="footer">
            <span onclick="openRetriever()">리트리버란?</span>
            <span onclick="showEmail()">문의 <span id="email" class="hidden-email">: dlwldnjst@gmail.com</span></span>
        </div>
        <script>
        function showEmail() {
            var emailElement = document.getElementById('email');
            emailElement.style.display = 'inline';
            var footerElement = document.querySelector('.footer span:last-child');
            footerElement.onclick = null; // 클릭 이벤트 제거
        }
        function openRetriever() {
            window.open('https://epik1.tistory.com/1', '_blank');
        }
        </script>
        """,
        height=50
    )

if __name__ == "__main__":
    main()
