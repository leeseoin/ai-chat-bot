import streamlit as st
import subprocess
import os
import re
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from uuid import uuid4
import pandas as pd
import json

# Ollama & Chroma 설정
LLM_BASE_URL = "http://localhost:11434"
llm = Ollama(model="llama3.2:1b", base_url=LLM_BASE_URL)
embeddings = OllamaEmbeddings(base_url=LLM_BASE_URL, model="llama3.2:1b")

# Chroma 클라이언트 설정 함수
def create_chroma_client(collection_name):
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

# Chroma 클라이언트 생성
chroma_client_pdf = create_chroma_client("pa_documents")
chroma_client_api_list = create_chroma_client("api_list")
chroma_client_api_spec = create_chroma_client("api_spec")
chroma_client_dbTable = create_chroma_client("dbTable")
chroma_client_puml = create_chroma_client("puml") 

# Streamlit UI 제목
st.title("Llama3.2: 1b 모델 탑재한 챗봇")

# 경로 설정
UPLOAD_DIR = "/proj/mini-chat-bot/chatbot/data_result"
os.makedirs(UPLOAD_DIR, exist_ok=True)
PY_SCRIPT_DIR = "/proj/mini-chat-bot/chatbot/python_script"
UML_ORIGINAL = os.path.join(UPLOAD_DIR, "UML_ORIGINAL") # puml
UML2IMG = os.path.join(UPLOAD_DIR, "UML2IMG") # puml
os.makedirs(UML_ORIGINAL, exist_ok=True) # puml
os.makedirs(UML2IMG, exist_ok=True) # puml

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None

# PDF 처리 함수
def process_pdf(file_path):
    pdf_file_name = os.path.basename(file_path)
    pdf_name_only = os.path.splitext(pdf_file_name)[0]

    # PDF별 PNG 저장 폴더 설정
    pdf_png_dir = os.path.join(UPLOAD_DIR, "DEVIDED_PDF_DIR", pdf_name_only)

    scripts = ["devide_pdf.py", "pdf2txt.py", "del_noWF.py", "pa_number.py"]

    with st.spinner(f"{pdf_file_name} 처리중..!"):
        for script in scripts:
            script_path = os.path.join(PY_SCRIPT_DIR, script)
            if not os.path.exists(script_path):
                st.error(f"{script} 경로가 존재하지 않습니다! ❌")
                continue

            result = subprocess.run(["python", script_path, file_path], capture_output=True, text=True)

            # 실행 결과 출력
            if result.returncode == 0:
                st.success(f"{script} 실행 완료 ✅")
            else:
                st.error(f"{script} 실행 중 오류 발생 ❌\n{result.stderr}")
                continue

    # PA 넘버 파일 경로 설정
    PA_FILE = os.path.join(UPLOAD_DIR, "EXTRACTED_ONLY_PA_NUMBER_EACHPAGE", f"{pdf_name_only}_pa_number.txt")

    # PA 넘버 파일이 존재하는지 확인
    if not os.path.exists(PA_FILE):
        st.error("PA 넘버 파일이 존재하지 않습니다! ❌")
        return []

    # 1. Chroma DB에 저장할 데이터 정리
    pa_mapping = {}

    with open(PA_FILE, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # PA 넘버 정보 매핑
    matches = re.findall(r'--- page_(\d+) ---\n(.*)', content)
    for match in matches:
        page_number = int(match[0])
        pa_number = match[1].strip()
        if pa_number.lower() != "none":  # PA넘버가 'None'인 결과값은 저장하지 않음
            pa_mapping[page_number] = pa_number

    # 2. PDF 이미지와 PA 넘버 매칭 후 Chroma DB 저장
    documents, ids, embeddings_list, metadatas = [], [], [], []

    for page_file in sorted(os.listdir(pdf_png_dir)):
        match = re.search(r'page_(\d+)\.png', page_file)
        if match:
            page_number = int(match.group(1))
            if page_number in pa_mapping:
                pa_number = pa_mapping[page_number]
                image_path = os.path.join(pdf_png_dir, page_file)

                # 문서 내용과 메타데이터 구성
                doc_content = f"PA 넘버: {pa_number}"
                doc_metadata = {
                    "pa_number": pa_number,
                    "image_path": image_path,
                    "pdf_filename": pdf_file_name
                }

                # ChromaDB 저장을 위한 데이터 리스트 구성
                documents.append(doc_content)
                metadatas.append(doc_metadata)
                ids.append(str(uuid4()))
                embeddings_list.append(embeddings.embed_query(doc_content))

    # 3. ChromaDB에 upsert 방식으로 저장
    if documents:
        chroma_client_pdf._collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents
        )

        # 세션 상태 업데이트
        st.session_state.processed_files.add(pdf_file_name)
        st.session_state.current_pdf = pdf_file_name
        st.success(f"✅ {len(documents)}개의 페이지 데이터가 ChromaDB에 저장되었습니다!")

    return list(pa_mapping.values())

# api_list 데이터 chroma db 적제
def store_api_list_in_chroma(excel_file_path):
    API_LIST_DIR = os.path.join(UPLOAD_DIR, "API_LIST_DIR")
    json_output_path = os.path.join(API_LIST_DIR, f"{os.path.splitext(os.path.basename(excel_file_path))[0]}.json")

    if not os.path.exists(json_output_path):
        st.error(f"API 리스트 JSON 파일을 찾을 수 없습니다: {json_output_path}")
        return False

    with open(json_output_path, "r", encoding="utf-8") as f:
        api_data = json.load(f)

    documents, metadatas, ids, embeddings_list = [], [], [], []

    for sheet in api_data["API리스트"]:
        for item in sheet["data"]:
            api_id = item.get("API ID", "") or "" # api_list.py의 결과물에서 API ID 추출(ex.CMM001)
            pa_number = item.get("사용화면아이디\n(없으면 비화면 API)", "") or "" # api_list.py의 결과물에서 사용화면아이디\n(없으면 비화면 API) 추출(ex.PA1201001)

            # 모든 키-값 쌍을 문자열로 변환하여 document에 추가
            doc_content = "\n".join([f"{k}: {v}" for k, v in item.items() if v is not None])

            documents.append(doc_content)
            metadatas.append({
                "source_file": excel_file_path,
                "sheet_name": sheet["sheet_name"],
                "api_id": api_id,
                "pa_number": pa_number
            })
            ids.append(str(uuid4()))
            embeddings_list.append(embeddings.embed_query(doc_content))

    # ChromaDB에 저장
    chroma_client_api_list._collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings_list
    )

    st.success(f"✅ API 리스트 {len(documents)}개가 ChromaDB에 저장되었습니다!")
    return True

# API명세서 데이터 chroma db 적제
def store_api_spec_in_chroma(excel_file_path):
    API_DIR = os.path.join(UPLOAD_DIR, "API_DIR")
    json_output_path = os.path.join(API_DIR, f"{os.path.splitext(os.path.basename(excel_file_path))[0]}.json")

    if not os.path.exists(json_output_path):
        st.error(f"API명세서 JSON 파일을 찾을 수 없습니다: {json_output_path}")
        return False

    with open(json_output_path, "r", encoding="utf-8") as f:
        api_spec_data = json.load(f)

    documents, metadatas, ids, embeddings_list = [], [], [], []
    for sheet_name, api_spec_list in api_spec_data.items(): # "API명세서"로 시작하는 시트이름으로 iteration
        for item in api_spec_list:
            # API ID 추출 (ex. item["설명"]["API ID"]
            api_id = item.get("설명", {}).get("API ID", "") or "" # api_specification.py의 결과물에서 "설명":{"API ID": "CMM001"} → "CMM001" 추출

            # document에는 API ID만 저장
            doc_content = f"API ID: {api_id}"

            # API 명세 정보를 JSON 문자열로 변환하여 저장
            api_spec_info = json.dumps(item, ensure_ascii=False)

            metadata = {
                "api_id": api_id, # API ID를 metadata에 추가
                "source_file": excel_file_path,
                "sheet_name": sheet_name,
                "api_spec_info": api_spec_info  # API 명세 정보를 JSON 문자열로 저장
            }

            documents.append(doc_content)
            metadatas.append(metadata)
            ids.append(str(uuid4()))
            embeddings_list.append(embeddings.embed_query(doc_content))
    # Chroma DB에 저장
    chroma_client_api_spec._collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings_list
    )

    st.success(f"✅ API 명세서 {len(documents)}개가 ChromaDB에 저장되었습니다!")
    return True

# 엑셀 파일 처리 및 ChromaDB 저장 함수
def process_excel(file_path):
    file_name = os.path.basename(file_path)

    if file_name in st.session_state.processed_files:
        st.error(f"이미 처리된 파일입니다: {file_name}")
        return False

    try:
        # 엑셀 파일 관련 스크립트 실행
        scripts = []
        xls = pd.ExcelFile(file_path)
        sheets = xls.sheet_names

        if any("DB_TABLE" in sheet for sheet in sheets):
            scripts.append("dbTable2json.py")
        if any("기능목록정의서" in sheet for sheet in sheets):
            scripts.append("fc.py")
        if any("API명세서" in sheet for sheet in sheets):
            scripts.append("api_specification.py")
        if any("API리스트" in sheet for sheet in sheets):
            scripts.append("api_list.py")

        for script in scripts:
            script_path = os.path.join(PY_SCRIPT_DIR, script)
            result = subprocess.run(
                ["python", script_path, file_path],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                st.success(f"{script} 실행 완료 ✅")
            else:
                st.error(f"{script} 실행 실패 ❌\n{result.stderr}")
                return False  # 스크립트 실패 시 즉시 종료

        # API 리스트 및 명세서 ChromaDB에 저장
        success_api_list = store_api_list_in_chroma(file_path)
        # API 명세서를 ChromaDB에 저장
        success_api_spec = store_api_spec_in_chroma(file_path)

        if success_api_list and success_api_spec:
            st.session_state.processed_files.add(file_name)
            st.success(f"✅ 엑셀 파일({file_name}) 처리 완료!")
            return True
        else:
            st.error("❌ 엑셀 파일의 API 리스트 또는 명세서 ChromaDB 저장 실패!")
            return False

    except Exception as e:
        st.error(f"❌ 엑셀 처리 중 오류 발생: {str(e)}")
        return False

# UML 파일 처리 및 업로드 함수
def process_puml(file_path):
    file_name = os.path.basename(file_path)
    if file_name in st.session_state.processed_files:
        st.error(f"이미 처리된 파일입니다!!: {file_name}")
        return False

    try:
        # puml 파일 관련 스크립트 실행
        script_path = os.path.join(PY_SCRIPT_DIR, "convert_uml2img.py")
        result = subprocess.run(
            ["python", script_path, file_path],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            # Extract the returned values from the standard output
            output_lines = result.stdout.strip().split('\n')
            png_path = None
            title_code = None
            db_tables = None

            for line in output_lines:
                if line.startswith("PNG Path:"):
                    png_path = line.split("PNG Path:")[1].strip()
                elif line.startswith("Title Code:"):
                    title_code = line.split("Title Code:")[1].strip()
                elif line.startswith("DB Tables:"):
                    db_tables = line.split("DB Tables:")[1].strip().replace("[","").replace("]","").replace("'","").split(", ")

            if png_path and title_code and db_tables:
                # Store data in ChromaDB
                doc_content = f"UML Diagram for API ID: {title_code}"
                metadata = {
                    "api_id": title_code,
                    "db_name": db_tables,
                    "png_path": png_path,
                    "source_file": file_name
                }
                chroma_client_puml._collection.upsert(
                    ids=[str(uuid4())],
                    documents=[doc_content],
                    metadatas=[metadata],
                    embeddings=[embeddings.embed_query(doc_content)]
                )

                st.success(f"✅ UML 파일({file_name}) 처리 완료! API ID: {title_code}, DB Tables: {db_tables}")
                st.session_state.processed_files.add(file_name)  # Mark as processed
                return True
            else:
                st.error("❌ UML 파일 처리 중 필요한 정보 추출 실패!")
                return False
        else:
            st.error(f"❌ convert_uml2img.py 실행 실패 ❌\n{result.stderr}")
            return False

    except Exception as e:
        st.error(f"❌ puml 처리 중 오류 발생: {str(e)}")
        return False

# 파일 업로드 섹션
uploaded_file = st.file_uploader("파일을 업로드하세요.(pdf, xlsx, puml)", type=["pdf", "xlsx", "puml"])

# 파일 업로드 후 처리
if uploaded_file:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension == "pdf":
        if uploaded_file.name not in st.session_state.processed_files:
            try:
                pa_numbers = process_pdf(file_path)
                if pa_numbers:
                    st.success(f"✅ 새로운 PDF 파일({uploaded_file.name})이 처리되었습니다!")
            except Exception as e:
                st.error(f"❌ PDF 처리 실패: {str(e)}")

    elif file_extension == "xlsx":
        success = process_excel(file_path)
        if not success:
            st.error("❌ 엑셀 처리 실패!")

    elif file_extension == "puml":
        success = process_puml(file_path)
        if not success:
            st.error("🚫 puml 파일 처리 실패!!")
    else:
        st.error("❌ 지원하지 않는 파일 형식입니다. (pdf, xlsx, puml 파일만 업로드 가능)")

# API 정보 검색 함수
def search_api_info(pa_number):
    query_embedding = embeddings.embed_query(pa_number)
    results = chroma_client_api_list._collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where={"pa_number": pa_number}
    )
    return results

# API 명세 정보 검색 함수
def search_api_spec_info(api_id):
    query_embedding = embeddings.embed_query(api_id)
    results = chroma_client_api_spec._collection.query(
        query_embeddings=[query_embedding],
        n_results=5,  
        where={"api_id": api_id}
    )
    return results

# UML 정보 검색 함수
def search_uml_info(api_id):
    results = chroma_client_puml._collection.query(
        query_embeddings=[embeddings.embed_query(api_id)],
        n_results=5,
        where={"api_id": api_id}
    )
    return results

# 채팅 UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력 처리
user_input = st.chat_input("질문을 입력하세요...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 1. PA 넘버 추출
    pa_number_match = re.search(r'PA\d+', user_input)
    pa_number = pa_number_match.group(0) if pa_number_match else None

    # 2. PDF 파일명 추출(".pdf"가 포함된 문자열)
    pdf_filename_match = re.search(r"([\w\-]+\.pdf)", user_input)
    pdf_filename = pdf_filename_match.group(1) if pdf_filename_match else None

    # API ID 추출 시도
    api_id_match = re.search(r'API ID: (\w+)', user_input)
    api_id = api_id_match.group(1) if api_id_match else None

    if pa_number_match and pdf_filename_match:
        pa_number = pa_number_match.group(0)
        pdf_filename = pdf_filename_match.group(1)

        # 3. ChromaDB에서 검색 (PA 넘버와 PDF 파일명이 일치하는 데이터 찾기)
        pdf_results = chroma_client_pdf._collection.query(
            query_embeddings=[embeddings.embed_query(pa_number)],
            n_results=10,
            where={
                "$and": [
                    {"pa_number": pa_number},
                    {"pdf_filename": pdf_filename}
                ]
            }
        )

        # 4. api_list에서 PA 넘버 관련 정보 검색
        api_list_results = search_api_info(pa_number)

        if api_list_results['documents'] and pdf_results['documents']:
            # api_list_info를 문자열로 결합하기 전에 데이터 구조 확인
            if isinstance(api_list_results['documents'][0], list):
                api_list_info = "\n".join(api_list_results['documents'][0])  # 첫 번째 리스트의 요소만 사용
            else:
                api_list_info = "\n".join(api_list_results['documents'])

            # 5. API ID 추출
            api_id_match = re.search(r'API ID: (\w+)', api_list_info)
            api_id = api_id_match.group(1) if api_id_match else None

            # 이미지 표시
            images = []
            for metadata in pdf_results['metadatas'][0]:
                image_path = metadata.get('image_path', '')
                if os.path.exists(image_path):
                    images.append(image_path)

            if images:
                for img in images:
                    st.image(img, caption=f"{pdf_filename}의 {pa_number}")

                # 6. api_spec에서 API ID로 정보 검색
                if api_id:
                    api_spec_results = search_api_spec_info(api_id)

                    # 디버깅으로 API Spec Results 출력
                    st.write("API Spec Results:", api_spec_results)
                    st.write("API Spec Results Metadatas:", api_spec_results['metadatas'])

                    if api_spec_results and 'metadatas' in api_spec_results and api_spec_results['metadatas']:
                        # metadatas가 이중 리스트 구조이므로 첫 번째 요소의 첫 번째 요소에 접근
                        first_metadata = api_spec_results['metadatas'][0][0]
                        if isinstance(first_metadata, dict) and 'api_spec_info' in first_metadata:
                            api_spec_info = first_metadata['api_spec_info']
                            try:
                                api_spec = json.loads(api_spec_info)
                                api_spec_str = json.dumps(api_spec, indent=4, ensure_ascii=False)
                            except json.JSONDecodeError:
                                api_spec_str = "API 명세 정보를 파싱하는 데 실패했습니다."
                        else:
                            api_spec_str = "API 명세 정보를 찾을 수 없습니다."
                    else:
                        api_spec_str = "API 명세 정보를 찾을 수 없습니다."

                    response = f"{pdf_filename}에서 PA넘버: {pa_number} 관련 이미지 {len(images)}개와 API 정보:\n{api_list_info}\n\nAPI 명세 정보:\n{api_spec_str}"

    # Check for API ID and search UML info
    if api_id:
        uml_results = search_uml_info(api_id)

        if uml_results and uml_results['metadatas']:
            uml_metadata = uml_results['metadatas'][0][0]  # Access the first element of the first list
            uml_image_path = uml_metadata.get('png_path', None)

            if uml_image_path and os.path.exists(uml_image_path):
                st.image(uml_image_path, caption=f"UML Diagram for API ID: {api_id}")
                response += f"\n\nUML Diagram:\n![UML Diagram]({uml_image_path})"  # Append to the response

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)