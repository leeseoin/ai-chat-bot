import os
from pdf2image import convert_from_path

# UPLOAD_DIR 경로 설정
UPLOAD_DIR = "/proj/mini-chat-bot/chatbot/data_result"
OUTPUT_BASE_DIR = os.path.join(UPLOAD_DIR, "DEVIDED_PDF_DIR")

def get_latest_pdf(directory):
    """디렉토리에서 가장 최근에 추가된 PDF 파일을 찾는 함수"""
    pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pdf")]
    
    if not pdf_files:
        raise FileNotFoundError("UPLOAD_DIR에 PDF 파일이 없습니다.")

    # 파일 수정 시간 기준으로 가장 최근 파일 찾기
    latest_pdf = max(pdf_files, key=os.path.getmtime)
    return latest_pdf

def divide_pdf(pdf_path):
    """PDF를 페이지별로 이미지로 변환하여 저장"""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {pdf_path}")

    # PDF 파일 이름 추출
    pdf_file_name = os.path.basename(pdf_path)
    pdf_name_only = os.path.splitext(pdf_file_name)[0]

    # PDF별 개별 디렉토리 생성
    output_dir = os.path.join(OUTPUT_BASE_DIR, pdf_name_only)
    os.makedirs(output_dir, exist_ok=True)

    # PDF 파일을 이미지로 변환
    images = convert_from_path(pdf_path, dpi=300)  # dpi 값으로 해상도 설정

    # 각 페이지를 이미지로 저장
    for page_number, image in enumerate(images, start=1):
        output_path = os.path.join(output_dir, f"page_{page_number}.png")
        image.save(output_path, "PNG")
        print(f"Saved: {output_path}")

    return output_dir  # 저장된 디렉토리 경로 반환

if __name__ == "__main__":
    try:
        latest_pdf = get_latest_pdf(UPLOAD_DIR)
        output_dir = divide_pdf(latest_pdf)
        print(f"Images for {os.path.basename(latest_pdf)} saved in: {output_dir}")
    except FileNotFoundError as e:
        print(e)
