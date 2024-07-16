import pandas as pd
import streamlit as st
import torch
import logging
import re
import time
from cachetools import cached, TTLCache
import google.generativeai as genai
import requests
from tornado.websocket import WebSocketClosedError
import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import traceback 
import psutil
import mysql.connector
from mysql.connector import Error
from io import BytesIO
import os

def load_model():
    return SentenceTransformer('xlm-r-bert-base-nli-stsb-mean-tokens')

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 3)  # Convert bytes to GB

def get_text_embedding(text):
    # 모델을 사용할 때만 로드
    model = load_model()
    embedding = model.encode(text)
    return np.array(embedding).flatten()  # 1D 배열로 변환

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

def connect_to_mariadb():
    try:
        connection = mysql.connector.connect(
            host=st.secrets["mariadb"]["DB_HOST"],
            user=st.secrets["mariadb"]["DB_USER"],
            password=st.secrets["mariadb"]["DB_PASSWORD"],
            database=st.secrets["mariadb"]["DB_NAME"]
        )
        if connection.is_connected():
            logger.info("MariaDB에 성공적으로 연결되었습니다.")
            return connection
    except Error as e:
        logger.error(f"MariaDB 연결 중 오류 발생: {e}")
        st.error(f"MariaDB 연결 중 오류 발생: {e}")
        return None

# 연결 확인 및 재연결 함수
def get_db_connection():
    if 'db_connection' not in st.session_state or st.session_state.db_connection is None:
        st.session_state.db_connection = connect_to_mariadb()
    else:
        try:
            if not st.session_state.db_connection.is_connected():
                st.session_state.db_connection = connect_to_mariadb()
        except Error as e:
            logger.error(f"연결 상태 확인 중 오류 발생: {e}")
            st.session_state.db_connection = connect_to_mariadb()
    return st.session_state.db_connection

def cleanup():
    if 'db_connection' in st.session_state and st.session_state.db_connection:
        st.session_state.db_connection.close()
    for key in st.session_state.keys():
        st.session_state[key] = None

atexit.register(cleanup)

def validate_and_process_excel(file):
    try:
        df = pd.read_excel(BytesIO(file.read()))
        
        required_columns = ['isbn', '청구기호']
        missing_columns = [col for col in required_columns if col.lower() not in [col.lower() for col in df.columns]]
        if missing_columns:
            return None, f"다음 필수 컬럼이 없습니다: {', '.join(missing_columns)}"
        
        def normalize_isbn(isbn):
            isbn = str(isbn).replace('-', '').replace(' ', '').lower()
            return isbn if isbn.isdigit() else None
        
        isbn_column = next(col for col in df.columns if col.lower() == 'isbn')
        call_number_column = next(col for col in df.columns if '청구기호' in col)
        
        df[isbn_column] = df[isbn_column].apply(normalize_isbn)
        df[call_number_column] = df[call_number_column].astype(str)
        
        df = df.dropna(subset=[isbn_column])
        df['processed_isbn'] = df[isbn_column]
        
        return df, None
    except Exception as e:
        return None, f"파일 처리 중 오류 발생: {str(e)}"

# 유저 입력을 전처리하는 함수
def preprocess_input(text):
    return re.sub(r'\s+', ' ', text).strip()

# API 호출 결과 캐싱
cache = TTLCache(maxsize=5, ttl=300)

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
    logger.info(f"Searching for keywords using LIKE: {keywords}")

    if not keywords:
        logger.warning("검색할 키워드가 없습니다.")
        return pd.DataFrame()

    # 데이터베이스 연결 확인
    if 'db_connection' not in st.session_state or st.session_state.db_connection is None:
        st.session_state.db_connection = connect_to_mariadb()
    
    if st.session_state.db_connection is None:
        logger.error("데이터베이스 연결이 없습니다.")
        st.session_state.error_message = "데이터베이스 연결이 없습니다."
        return pd.DataFrame()

    cursor = st.session_state.db_connection.cursor(dictionary=True)

    like_clauses = " OR ".join(["title LIKE %s", "author LIKE %s", "description LIKE %s"] * len(keywords))
    query = f"""
    SELECT title, author, description, isbn
    FROM books
    WHERE {like_clauses}
    """

    like_values = [f"%{keyword}%" for keyword in keywords for _ in range(3)]
    logger.debug(f"Executing query: {query} with like_values: {like_values}")

    try:
        cursor.execute(query, like_values)
        results = cursor.fetchall()
        logger.info(f"검색 결과 수: {len(results)}")

        if not results:
            st.session_state.warning_message = "검색 결과가 없습니다."
            logger.warning("검색 결과가 없습니다.")

        df = pd.DataFrame(results).drop_duplicates()
        logger.info(f"중복 제거 후 검색 결과 DataFrame 크기: {df.shape}")
        return df
    except mysql.connector.Error as e:
        logger.error(f"MariaDB 검색 중 오류: {e}")
        st.session_state.error_message = f"데이터베이스 검색 중 오류 발생: {str(e)}"
        return pd.DataFrame()

def check_db_connection(conn):
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        return result is not None
    return False


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

    # ISBN 및 청구기호 컬럼 확인
    isbn_col = next((col for col in user_books.columns if 'isbn' in col.lower()), None)
    call_number_col = next((col for col in user_books.columns if '청구기호' in col), None)

    if not isbn_col or not call_number_col:
        logger.error(f"ISBN 또는 청구기호 컬럼을 찾을 수 없습니다. ISBN 컬럼: {isbn_col}, 청구기호 컬럼: {call_number_col}")
        return pd.DataFrame()

    # 검색 결과 및 사용자 도서 ISBN 전처리
    search_results['processed_isbn'] = search_results['isbn'].astype(str).str.replace('-', '').str.lower()
    user_books['processed_isbn'] = user_books[isbn_col].astype(str).str.replace('-', '').str.lower()

    # 검색 결과와 사용자 도서를 병합
    merged = pd.merge(search_results, user_books[['processed_isbn', call_number_col]], on='processed_isbn', how='left')

    # 디버깅: 병합된 데이터프레임의 컬럼명 확인
    logger.debug(f"Merged DataFrame columns: {merged.columns}")

    # 청구기호가 있는 도서 필터링
    if call_number_col not in merged.columns:
        logger.error(f"병합된 데이터프레임에 청구기호 컬럼이 없습니다. 현재 컬럼들: {merged.columns}")
        return pd.DataFrame()
    
    filtered_books = merged.dropna(subset=[call_number_col])
    filtered_books.rename(columns={call_number_col: '청구기호'}, inplace=True)

    # ISBN을 기준으로 중복 제거
    filtered_books = filtered_books.drop_duplicates(subset=['processed_isbn'])

    logger.info(f"총 도서 수: {len(merged)}, 청구기호가 있는 도서 수: {len(filtered_books)}")
    st.dataframe(filtered_books)
    
    return filtered_books if not filtered_books.empty else pd.DataFrame()

def get_book_recommendations(user_input, filtered_books):
    if filtered_books is None or filtered_books.empty:
        return "죄송합니다. 요청하신 조건에 맞는 도서를 찾지 못했습니다."

    book_list = "\n".join([
        f"ISBN: {book['isbn']}, 제목: {book['title']}, 저자: {book['author']}, 청구기호: {book['청구기호']}, 설명: {book['description']}"
        for _, book in filtered_books.iterrows()
    ])
    prompt = f"""
    사용자의 요청에 맞는 책을 추천해주세요. 다음 도서 목록을 바탕으로 최소 1권에서 최대 5권까지 추천해주세요.
    각 책은 최소 5문장으로 다음 내용을 포함해야 합니다:
    - 주요 내용 요약
    - 핵심 주제나 사건
    - 책의 구성이나 특징
    - 독자에게 줄 수 있는 통찰이나 가치
    - 추천 이유

    조건:
    1. 목록에 없는 책은 추천하지 마세요.
    2. 제목이나 저자를 변경하지 마세요.
    3. 시리즈는 1권만 추천하세요.
    4. 한글로만 답해주세요.
    5. 형식:
        [BOOK_START]
        ISBN: [ISBN]
        제목: [제목]
        저자: [저자]
        청구기호: [청구기호]
        설명: [설명]
        [BOOK_END]

    도서 목록:
    {book_list}

    사용자 요청: {user_input}
    """

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

@st.cache_resource
def load_model():
    return SentenceTransformer('xlm-r-bert-base-nli-stsb-mean-tokens')

model = load_model()

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

def process_user_input_sync(user_input):
    processed_input = preprocess_input(user_input)
    
    try:
        with st.spinner('리트리버가 책을 찾고 있어요...'):
            # Keyword extraction
            st.write("1. 키워드 추출 중...")
            extracted_keywords = extract_keywords(processed_input)
            st.session_state.keywords = extracted_keywords
            st.write(f"추출된 키워드: {st.session_state.keywords}")
            
            if not st.session_state.keywords:
                st.warning("키워드를 추출하지 못했습니다. 다른 질문을 시도해보세요.")
                return

            # Book search
            st.write("2. 키워드로 도서 검색 중...")
            search_results = search_books(st.session_state.keywords)
            
            if search_results is None or search_results.empty:
                st.warning("검색 결과가 없습니다. 다른 키워드로 시도해보세요.")
                logger.warning(f"No search results found for keywords: {st.session_state.keywords}")
                return
            
            st.write(f"검색된 도서 수: {len(search_results)}")

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
                top_books = search_with_vector(st.session_state.filtered_books, ' '.join(st.session_state.keywords), top_k=10)
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
        st.subheader("리트리버 추천 도서")
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

def extract_book_info(book_text):
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
    
    return book if book else None

def parse_response(response):
    if "죄송합니다" in response or "추천할 수 없습니다" in response:
        logger.warning("API가 추천을 제공하지 못했습니다.")
        return []

    book_pattern = r'\[BOOK_START\](.*?)\[BOOK_END\]'
    book_matches = re.findall(book_pattern, response, re.DOTALL)
    
    if not book_matches:
        logger.warning("책 정보를 파싱하지 못했습니다.")
        return []

    books = []
    for book_text in book_matches:
        book = extract_book_info(book_text)
        if book:
            books.append(book)
    
    if not books:
        logger.warning("책 정보를 파싱하지 못했습니다.")
    
    return books

def fetch_single_book_info(book, headers, api_url, user_books_df):
    isbn = book.get('isbn', '').replace(',', '').replace(' ', '').lower()
    if not isbn:
        return None, f"ISBN이 없는 책 정보가 있습니다: {book}"

    params = {"target": "isbn", "query": isbn}
    try:
        response = requests.get(api_url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if data["documents"]:
                return process_api_response(data, book, isbn, user_books_df)
            else:
                return None, f"API에서 ISBN {isbn}에 대한 정보를 찾지 못했습니다."
        else:
            return None, f"API 요청 실패 (상태 코드: {response.status_code})"
    except Exception as e:
        return None, f"ISBN: {isbn}에 대한 요청 중 예외가 발생했습니다: {e}"

def process_api_response(data, book, isbn, user_books_df):
    api_book_info = data["documents"][0]
    
    matching_row = user_books_df[user_books_df['processed_isbn'].str.contains(isbn, na=False)]
    if matching_row.empty:
        matching_row = user_books_df[user_books_df['processed_isbn'].str.endswith(isbn[-9:], na=False)]
    
    call_number = matching_row['청구기호'].values[0] if not matching_row.empty else 'N/A'
    
    if call_number == 'N/A':
        return None, f"ISBN {isbn}에 대한 청구기호를 찾지 못했습니다."

    return {
        "title": book.get('title', api_book_info.get('title', '')),
        "author": book.get('author', api_book_info.get('authors', [''])[0]),
        "contents": book.get('contents', api_book_info.get('contents', '')),
        "isbn": isbn,
        "thumbnail": api_book_info.get("thumbnail", ""),
        "call_number": call_number
    }, None

def fetch_book_info_batch(books, headers, api_url, user_books_df):
    if not books:
        logger.error("책 정보 리스트가 비어 있습니다.")
        return pd.DataFrame()

    book_info_list = []
    warnings = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_book = {executor.submit(fetch_single_book_info, book, headers, api_url, user_books_df): book for book in books}
        for future in as_completed(future_to_book):
            book_info, warning_message = future.result()
            if book_info:
                book_info_list.append(book_info)
            if warning_message:
                warnings.append(warning_message)
            time.sleep(0.1)  # API rate limiting을 위한 딜레이

    for warning in warnings:
        st.warning(warning)

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
def search_with_vector(filtered_books, query, top_k=10):
    try:
        keywords = query.split()
        search_results = search_books(keywords)
        
        if search_results.empty:
            st.write("검색 결과가 없습니다.")
            logger.error("검색 결과가 없습니다.")
            return pd.DataFrame()

        search_results = search_results.head(top_k)
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

        top_results = search_results.iloc[indices[0]].drop_duplicates()
        logger.debug(f"중복 제거 후 Top results: {top_results}")

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
    for attempt in range(retries):
        try:
            response = model.generate_content(contents=prompt)
            if response.text:
                return response.text
            elif hasattr(response, 'prompt_feedback'):
                logger.warning(f"API response blocked: {response.prompt_feedback}")
                return "죄송합니다. 요청하신 조건에 맞는 도서를 찾지 못했습니다."
            else:
                logger.warning("API response is empty")
                return "죄송합니다. 요청하신 조건에 맞는 도서를 찾지 못했습니다."
        except Exception as e:
            logger.error(f"API call attempt {attempt + 1} failed: {str(e)}")
            if attempt < retries - 1:
                time.sleep(wait_time)
    return "죄송합니다. 요청하신 조건에 맞는 도서를 찾지 못했습니다."

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

    st.title("리트리버: 책을 찾아줘 0.1 ver")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        user_books_file = st.file_uploader("소장 도서 목록 업로드", type=["xlsx"], key="excel_uploader")
        st.markdown("</div>", unsafe_allow_html=True)

        if user_books_file is not None:
            df, error = validate_and_process_excel(user_books_file)
            if error:
                st.error(error)
            else:
                st.session_state.user_books = df
                st.success('소장 도서 목록을 성공적으로 업로드하고 처리했습니다.')
                st.write("처리된 데이터 미리보기:")
                st.write(df.head())

                st.markdown('<div class="input-container">', unsafe_allow_html=True)
                with st.form(key='chat_form', clear_on_submit=True):
                    user_input = st.text_input("질문을 입력해 주세요:", key="user_input_form")
                    submit_button = st.form_submit_button(label='검색')

                if submit_button and user_input:
                    try:
                        results = process_user_input_sync(user_input)
                        st.session_state.filtered_books = pd.DataFrame()
                        st.session_state.recommendations = ""
                    except WebSocketClosedError:
                        st.error("웹소켓 연결이 닫혔습니다. 다시 시도해 주세요.")
        else:
            st.info("소장 도서 목록(엑셀)을 업로드해 주세요.")

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
            <br>
            <br>
            <span>ⓒ 2024. LEEJIWON. All rights reserved.</span>
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
        height=60
    )

    # CPU 사용량
    cpu_usage = psutil.cpu_percent(interval=1)
    st.metric(label="CPU Usage", value=f"{cpu_usage} %")

    # 메모리 사용량
    memory_info = psutil.virtual_memory()
    used_memory = memory_info.used / (1024 ** 3)  # 기가바이트 단위로 변환
    total_memory = memory_info.total / (1024 ** 3)  # 기가바이트 단위로 변환
    available_memory = memory_info.available / (1024 ** 3)  # 기가바이트 단위로 변환
    st.metric(label="Memory Usage", value=f"{used_memory:.2f} GB / {total_memory:.2f} GB", delta=f"{available_memory:.2f} GB available")

    # 디스크 사용량
    disk_usage = psutil.disk_usage('/')
    st.metric(label="Disk Usage", value=f"{disk_usage.percent} %")

    # 네트워크 사용량
    net_io = psutil.net_io_counters()
    st.metric(label="Network Sent", value=f"{net_io.bytes_sent / (1024 ** 2):.2f} MB")
    st.metric(label="Network Received", value=f"{net_io.bytes_recv / (1024 ** 2):.2f} MB")

    memory_usage = get_memory_usage()
    total_memory = 1.0  # Streamlit 클라우드 무료 플랜의 메모리 제한은 1GB
    available_memory = total_memory - memory_usage

    st.metric(label="Memory Usage", value=f"{memory_usage:.2f} GB / {total_memory:.2f} GB")
    st.metric(label="Available Memory", value=f"{available_memory:.2f} GB")

if __name__ == "__main__":
    main()
