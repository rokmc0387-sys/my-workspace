import streamlit as st
import streamlit.components.v1 as components
import webbrowser
import pandas as pd
import google.generativeai as genai
import time
import re
import json
import os
import numpy as np
import io
import concurrent.futures
import threading
import random
import pickle
import hashlib
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from crawlers_v1 import YouTubeCrawler, CoupangParser, AmazonParser, SamsungCrawler, LGCrawler, BestBuyParser

# ==========================================\\
# [설정] 페이지 설정
# ==========================================
st.set_page_config(
    page_title="VoC AI 분석기 (Gemini Powered)",
    page_icon="📊",
    layout="wide"
)

# 설정 파일 경로
CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".voc_app_config.json")

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {}

def save_config(config):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Config save failed: {e}")

def load_previous_work(base_dir):
    """지정된 경로에서 기존 작업 파일들을 찾아 세션 상태에 로드합니다."""
    if not os.path.exists(base_dir):
        return # 경로가 없으면 아무것도 안 함(빠져나옴)

    # 0. 수집 데이터
    collected_path = os.path.join(base_dir, "collected_data.csv")
    if os.path.exists(collected_path) and 'df_collected' not in st.session_state:
        try:
            st.session_state['df_collected'] = pd.read_csv(collected_path, on_bad_lines='skip', encoding='utf-8-sig')
            st.toast(f"✅ 수집된 데이터 로드 완료 ({len(st.session_state['df_collected'])}건)")
        except Exception as e:
            st.error(f"수집 데이터 로드 실패: {e}")

    # 1. 정제 데이터
    clean_path = os.path.join(base_dir, "cleaned_data.csv")
    if os.path.exists(clean_path) and 'df_cleaned' not in st.session_state:
        try:
            st.session_state['df_cleaned'] = pd.read_csv(clean_path, on_bad_lines='skip', encoding='utf-8-sig')
            st.toast(f"✅ 정제된 데이터 로드 완료 ({len(st.session_state['df_cleaned'])}건)")
        except Exception as e:
            st.error(f"정제 데이터 로드 실패: {e}")

    # 2. 추출 데이터
    extract_path = os.path.join(base_dir, "voc_results.csv")
    if os.path.exists(extract_path) and 'df_extracted' not in st.session_state:
        try:
            st.session_state['df_extracted'] = pd.read_csv(extract_path, on_bad_lines='skip', encoding='utf-8-sig')
            st.toast(f"✅ 추출된 데이터 로드 완료 ({len(st.session_state['df_extracted'])}건)")
        except Exception as e:
            st.error(f"추출 데이터 로드 실패: {e}")

    # 3. 전처리(Explode) 데이터
    explode_path = os.path.join(base_dir, "result_python_final.csv")
    if os.path.exists(explode_path) and 'df_exploded' not in st.session_state:
        try:
            st.session_state['df_exploded'] = pd.read_csv(explode_path, on_bad_lines='skip', encoding='utf-8-sig')
            st.toast(f"✅ 전처리 데이터 로드 완료 ({len(st.session_state['df_exploded'])}건)")
        except Exception as e:
            st.error(f"전처리 데이터 로드 실패: {e}")

    # 4. 수렴(Convergence) 데이터
    conv_path = os.path.join(base_dir, "AOS_converged_result.csv")
    if os.path.exists(conv_path) and 'df_final' not in st.session_state:
        try:
            st.session_state['df_final'] = pd.read_csv(conv_path, on_bad_lines='skip', encoding='utf-8-sig')
            st.toast(f"✅ 수렴 결과 데이터 로드 완료 ({len(st.session_state['df_final'])}건)")
        except Exception as e:
            st.error(f"수렴 데이터 로드 실패: {e}")

st.title("📊 고객 인사이트 분석 AI 에이전트")
st.markdown("""
 **AI + 개발 알고리즘**를 활용하여 고객 리뷰(VoC)를 분석합니다.
**수집 -> 정제 -> 추출 -> 전처리 -> 수렴 -> 시각화 -> M코드 동기화** 프로세스 지원,작업 내용은 **자동으로 저장/로드** 됩니다.
""")

# ==========================================
# [사이드바] 설정 및 상태
# ==========================================
with st.sidebar:
    st.header("⚙️ 설정")
    
    # 설정 로드
    config = load_config()
    default_dir = config.get("base_dir", r"C:\Users\user\n8n-data\VD 개선본")

    # API 키 입력 (Gemini)
    api_key = st.text_input("Gemini API Key", type="password", help="Google AI Studio에서 발급받은 API 키를 입력하세요.")
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
        genai.configure(api_key=api_key)
        st.success("Gemini API 키가 설정되었습니다!")
    else:
        st.warning("Gemini API 키를 입력해주세요.")

    st.divider()
    st.subheader("📁 작업 경로")
    # 기본 경로 설정
    base_dir = st.text_input("작업 폴더 경로", value=default_dir)
    
    if st.button("경로 설정 및 데이터 로드") or base_dir != st.session_state.get('last_loaded_dir'):
        if os.path.exists(base_dir):
            keys_to_clear = ['df_collected', 'df_cleaned', 'df_extracted', 'df_exploded', 'df_final']
            for k in keys_to_clear:
                if k in st.session_state:
                    del st.session_state[k]

            # 설정 저장
            if base_dir != config.get("base_dir"):
                save_config({"base_dir": base_dir})
            
            # 데이터 로드
            load_previous_work(base_dir)
            st.session_state['last_loaded_dir'] = base_dir
            st.success(f"작업 경로가 설정되었습니다: {base_dir}")
        else:
            st.error("존재하지 않는 경로입니다.")
            
    st.divider()
    st.markdown("### ⚡ 성능 설정")
    max_workers = st.slider(
        "동시 요청 수 (Concurrency)", 
        min_value=1, 
        max_value=20, 
        value=8, 
        help="동시에 처리할 API 요청 수입니다. 권장: 5~10 (Free Tier)",
        key="concurrency_slider"
    )

# ==========================================
# [함수] 공통 유틸리티 및 로직
# ==========================================

def save_and_accumulate(df_new, save_path):
    """새로운 데이터를 기존 파일에 누적(Accumulate)하여 저장하는 공통 함수"""
    if os.path.exists(save_path):
        try:
            df_prev = pd.read_csv(save_path)
            df_combined = pd.concat([df_prev, df_new], ignore_index=True)
        except Exception as e:
            st.error(f"기존 파일 병합 중 오류: {e}")
            df_combined = df_new
    else:
        df_combined = df_new

    # 중복 제거 (내용이 같으면 제거)
    if 'contents' in df_combined.columns:
        df_combined.drop_duplicates(subset=['contents'], keep='first', inplace=True)
    else:
        df_combined.drop_duplicates(inplace=True)

    # 인덱스 재설정
    df_combined['Index'] = range(1, len(df_combined) + 1)
    
    # 저장
    try:
        df_combined.to_csv(save_path, index=False, encoding='utf-8-sig')
        st.session_state['df_collected'] = df_combined
        st.toast(f"✅ 데이터 저장 및 누적 완료! (총 {len(df_combined)}건)")
    except Exception as e:
        st.error(f"파일 저장 실패: {e}")

# --- 1단계: 정제 로직 ----
def is_valid_sentence(text, irrelevant_keywords):
    if pd.isna(text): return False, "Null 값"
    text = str(text).strip()
    
    # 1. 기호/숫자로만 구성 (한글/영문 없음)
    if not re.search(r'[가-힣a-zA-Z]', text): return False, "무의미한 텍스트 (자음/이모티콘 등)"
    
    # 2. 단순 문장 (30자 미만)
    if len(text) < 20: return False, "20자 미만 (단순 문장)"
        
    # 3. 단순 질문 (30자 미만 & ?)
    if len(text) < 20 and '?' in text: return False, "20자 미만 단순 질문"
    
    # 4. 키워드 기반 제거 (v6) - Case insensitive phrasing
    # "OP"는 대문자이거나 단어 단위일 때만 제거 (오인 방지)
    if re.search(r'\bOP\b', text): return False, "제외 키워드 포함(OP)"
    
    text_lower = text.lower()
    v6_phrases = [
        "i agree", "i disagree", "good point", "you are right", "this is the way",
        "this post", "this comment", "thanks for sharing", "thank you for the info"
    ]
    
    for phrase in v6_phrases:
        if phrase in text_lower: return False, f"제외 키워드 포함({phrase})"
    
    # 기존 키워드 체크 (UI에서 넘어온 값)
    for keyword in irrelevant_keywords:
        if keyword in text: return False, f"제외 키워드 포함({keyword})"
    return True, "Pass"

# --- 2단계: 추출 로직 (KeywordExtractor) ----
class KeywordExtractor:
    """VoC 텍스트에서 Aspect, Opinion, Sentiment 키워드를 추출하는 클래스."""
    
    # ─── 클래스 상수 ───
    MODEL_NAME = "gemini-2.5-flash"
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_TOP_P = 0.8
    DEFAULT_TOP_K = 40
    DEFAULT_MAX_TOKENS = 8192
    MIN_TEXT_LENGTH = 2

    def __init__(self, api_key: str, level1_categories: list):
        """
        KeywordExtractor 초기화.
        
        Args:
            api_key: Gemini API 키
            level1_categories: 허용된 상위 범주(Aspect Lv1) 리스트
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.MODEL_NAME)
        self.level1_categories = level1_categories
        self.level1_categories_str = ", ".join(self.level1_categories)
        
        self.generation_config = {
            'temperature': self.DEFAULT_TEMPERATURE,
            'top_p': self.DEFAULT_TOP_P,
            'top_k': self.DEFAULT_TOP_K,
            'max_output_tokens': self.DEFAULT_MAX_TOKENS,
            'response_mime_type': 'application/json',
            'response_schema': self._get_response_schema()
        }
        
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

    @staticmethod
    def _get_response_schema() -> dict:
        """Gemini API 응답 스키마를 반환합니다."""
        return {
            "type": "OBJECT",
            "properties": {
                "reasoning": {"type": "STRING", "description": "분석 내용 및 근거 요약"},
                "keywords": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "aspect_lv1": {"type": "STRING"},
                            "aspect_lv2": {"type": "STRING"},
                            "opinion": {"type": "STRING"},
                            "sentiment": {"type": "STRING"}
                        },
                        "required": ["aspect_lv1", "aspect_lv2", "opinion", "sentiment"]
                    }
                }
            },
            "required": ["reasoning", "keywords"]
        }

    def clean_text(self, text: str) -> str:
        """입력 텍스트를 정제합니다."""
        if not isinstance(text, str):
            text = str(text)
        return text.strip()

    def validate_text(self, text: str) -> bool:
        """텍스트가 분석 가능한지 검증합니다."""
        return bool(text and len(text.strip()) >= self.MIN_TEXT_LENGTH)

    def _build_prompt(self, text: str) -> str:
        """VoC 분석용 프롬프트를 생성합니다."""
        return f'''당신은 정밀한 VoC 분석 전문가입니다. 
다음 텍스트에서 제품(TV, 모니터, 가전, IT기기, PC부품 등)과 관련된 핵심 정보를 분석하여 JSON으로 출력하세요.

분석 대상: "{text}"

[분석 가이드]
1. **Reasoning (추론):** 먼저 텍스트의 주요 내용, 감정, 기술적 맥락(스펙, 비교, 이슈 등)을 분석하여 `reasoning` 필드에 요약해 넣으세요.
2. **Extraction (추출):** 그 후, `keywords` 리스트에 다음 규칙에 따라 데이터를 추출하세요.

[추출 규칙]
1. `aspect_lv1`: 반드시 다음 중 하나 선택 -> [{self.level1_categories_str}]
   - 리스트에 없으면 무조건 "기타"
2. `aspect_lv2`: 구체적인 속성 (예: 주사율, 발열, AS, 가격 등)
3. `opinion`: 소비자의 구체적인 평가나 의견 (단순 형용사보다는 문맥을 살릴 것)
4. `sentiment`: "긍정", "부정", "중립" 중 하나
5. **적극적 추출:** IT/기술/구매 관련 내용(모니터, 그래픽카드, 설정, 가격, 비교 등)이 조금이라도 있다면 **반드시 추출**하세요. "관련 없음"으로 처리하지 마세요.

[예시 데이터]
입력: "5080 쓰시는 분들은 QHD 고주사율이 유리함. 96만원이었나. 그래도 AS 좋음"
출력 JSON 구조:
{{
  "reasoning": "사용자는 5080 그래픽카드와 QHD 모니터의 조합을 추천하며 가격과 AS를 언급함.",
  "keywords": [
    {{"aspect_lv1": "화질.디스플레이", "aspect_lv2": "주사율", "opinion": "QHD 고주사율이 유리함", "sentiment": "긍정"}},
    {{"aspect_lv1": "가격", "aspect_lv2": "가격", "opinion": "96만원 언급", "sentiment": "중립"}},
    {{"aspect_lv1": "서비스", "aspect_lv2": "AS", "opinion": "AS 좋음", "sentiment": "긍정"}}
  ]
}}'''

    def _parse_response(self, response) -> dict:
        """API 응답을 파싱하여 결과 딕셔너리를 반환합니다."""
        try:
            text_resp = response.text.strip()
            result = json.loads(text_resp)
            keywords = result.get("keywords", [])
            return {"success": True, "keywords": keywords}
        except json.JSONDecodeError:
            print(f"[JSON Error] Raw output: {response.text[:100]}...")
            return {"success": False, "keywords": []}

    def extract_keywords_with_retry(self, text: str, product_category: str = "VD", 
                                     product_model: str = "TV", max_retries: int = 3) -> dict:
        """재시도 로직이 포함된 키워드 추출."""
        delay = 1
        for attempt in range(max_retries):
            try:
                res = self.extract_keywords(text, product_category, product_model)
                if res.get("success"):
                    return res
            except Exception as e:
                print(f"Retry {attempt+1} failed: {e}")
            
            time.sleep(delay + random.uniform(0, 1))
            delay *= 2
            
        return {"success": False, "error": "Max retries exceeded", "keywords": []}

    def extract_keywords(self, text: str, product_category: str = "VD", 
                         product_model: str = "TV") -> dict:
        """텍스트에서 키워드를 추출합니다."""
        cleaned_text = self.clean_text(text)
        if not self.validate_text(cleaned_text):
            return {"success": False, "keywords": []}
        
        prompt = self._build_prompt(cleaned_text)

        try:
            response = self.model.generate_content(
                prompt, 
                generation_config=self.generation_config, 
                safety_settings=self.safety_settings
            )
            return self._parse_response(response)
            
        except Exception as e:
            if "429" in str(e) or "Resource exhausted" in str(e):
                raise e
            print(f"[API Error] {str(e)[:200]}")
            return {"success": False, "error": str(e), "keywords": []}


def preprocess_keywords(keywords_data, valid_categories=None):
    if not keywords_data: return "", "", "", ""
    final_aspect_lv1_list, final_aspect_lv2_list, final_opinion_list, final_sentiment_list = [], [], [], []
    
    for item in keywords_data:
        def split_and_clean(val):
            return [word.strip() for word in str(val).split('|') if word.strip()]

        lv1_raw = str(item.get('aspect_lv1', '기타'))
        lv2_raw = str(item.get('aspect_lv2', ''))
        op_raw = str(item.get('opinion', ''))
        sent_raw = str(item.get('sentiment', '중립'))

        lv1_splits = split_and_clean(lv1_raw)
        lv2_splits = split_and_clean(lv2_raw)
        op_splits = split_and_clean(op_raw)
        sent_splits = split_and_clean(sent_raw)

        if valid_categories:
            validated_lv1 = []
            valid_set = set(c.strip() for c in valid_categories)
            for cat in lv1_splits:
                cat_clean = cat.strip()
                if cat_clean in valid_set:
                    validated_lv1.append(cat_clean)
                else:
                    cat_tokens = set(cat_clean.replace('.', ' ').split())
                    matched_cat = next((vc for vc in valid_categories if cat_tokens & set(vc.replace('.', ' ').split())), '기타')
                    validated_lv1.append(matched_cat)
            lv1_splits = validated_lv1

        max_len = max(len(lv1_splits), len(lv2_splits), len(op_splits), len(sent_splits), 1)

        if not lv1_splits: lv1_splits = ['기타']
        while len(lv1_splits) < max_len: lv1_splits.append(lv1_splits[-1])
        if not lv2_splits: lv2_splits = [lv1_splits[0]]
        while len(lv2_splits) < max_len: lv2_splits.append(lv2_splits[-1])
        if not op_splits: op_splits = ['-']
        while len(op_splits) < max_len: op_splits.append(op_splits[-1])
        if not sent_splits: sent_splits = ['중립']
        while len(sent_splits) < max_len: sent_splits.append(sent_splits[-1])

        final_aspect_lv1_list.extend(lv1_splits[:max_len])
        final_aspect_lv2_list.extend(lv2_splits[:max_len])
        final_opinion_list.extend(op_splits[:max_len])
        final_sentiment_list.extend(sent_splits[:max_len])
    
    return "|".join(final_aspect_lv1_list), "|".join(final_aspect_lv2_list), "|".join(final_opinion_list), "|".join(final_sentiment_list)

# ==================================================================
# [NEW] Hybrid RAG: Vector Similarity Search + AI Few-shot
# ==================================================================
# 텍스트 깔끔히 청소 정리
def clean_text(text):
    """텍스트 정규화"""
    if not isinstance(text, str): return ""
    return re.sub(r'\s+', ' ', text).strip()

# --------------------------------------------------------------------------------
# 1단계: 벡터 DB 구축 (Gemini Embedding API)
# --------------------------------------------------------------------------------
def build_vector_index(df_embedding, api_key=None, progress_callback=None, max_workers=20):
    """
    Embedding Matrix를 벡터 DB로 변환 (로컬 임베딩 모델 사용)
    **sentence-transformers로 60배 빠르고 20% 더 정확!**
    
    Args:
        df_embedding: Embedding Matrix DataFrame
        api_key: 사용하지 않음 (하위 호환성 유지)
        progress_callback: 진행률 표시 콜백 함수 (optional)
        max_workers: 사용하지 않음 (하위 호환성 유지)
    
    Returns:
        (vector_db, metadata): 벡터 배열과 메타데이터 리스트
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers가 설치되지 않았습니다.\n"
            "설치: pip install sentence-transformers"
        )
    
    # 한국어 특화 모델 로드 (최초 1회만 다운로드, 이후 캐싱)
    model = SentenceTransformer('BM-K/KoSimCSE-roberta')
    
    total = len(df_embedding)
    
    # 데이터 준비
    all_texts = []
    all_metadata = []
    
    for idx, row in df_embedding.iterrows():
        aspect_text = clean_text(str(row.get('Aspect_Lv2', '')))
        opinion_text = clean_text(str(row.get('Opinion', '')))
        combined_text = f"Aspect: {aspect_text}, Opinion: {opinion_text}"
        
        all_texts.append(combined_text)
        all_metadata.append({
            'Aspect.수렴': clean_text(str(row.get('Aspect.수렴', ''))),
            'Opinion.수렴': clean_text(str(row.get('Opinion.수렴', ''))),
            'original_aspect': aspect_text,
            'original_opinion': opinion_text
        })
    
    # 배치 임베딩 (한 번에 모두 처리, 매우 빠름!)
    if progress_callback:
        progress_callback(0.1)
    
    # 배치 크기 32로 처리 (메모리 효율적)
    embeddings = model.encode(
        all_texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    
    if progress_callback:
        progress_callback(0.9)
    
    # NumPy 배열로 변환
    vector_db = np.array(embeddings)
    
    if progress_callback:
        progress_callback(1.0)
    
    return vector_db, all_metadata

# --------------------------------------------------------------------------------
# 2단계: 유사도 검색 (Cosine Similarity)
# --------------------------------------------------------------------------------


def search_similar_examples(query_aspect, query_opinion, vector_db, metadata, api_key=None, top_k=50, threshold=0.7, model=None):
    """
    입력과 가장 유사한 예제를 벡터 DB에서 검색 (로컬 모델 사용)
    
    Args:
        query_aspect: 검색할 Aspect_Lv2
        query_opinion: 검색할 Opinion
        vector_db: 벡터 DB (numpy array)
        metadata: 메타데이터 리스트
        api_key: 사용하지 않음 (하위 호환성 유지)
        top_k: 반환할 최대 예제 수 (기본 50)
        threshold: 최소 유사도 임계값 (기본 0.7)
        model: sentence-transformers 모델 (optional, 없으면 자동 로드)
    
    Returns:
        유사한 예제 리스트 (유사도 순 정렬)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return []
    
    # 모델 로드 (전달되지 않으면 새로 로드)
    if model is None:
        model = SentenceTransformer('BM-K/KoSimCSE-roberta')
    
    # 쿼리 텍스트 생성
    query_text = f"Aspect: {clean_text(query_aspect)}, Opinion: {clean_text(query_opinion)}"
    
    try:
        # 쿼리 벡터화 (로컬 모델)
        query_vector = model.encode([query_text], convert_to_numpy=True)
        
        # Cosine Similarity 계산
        similarities = cosine_similarity(query_vector, vector_db)[0]
        
        # Top-K 추출 (유사도 순 정렬)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] >= threshold:
                results.append({
                    'similarity': float(similarities[idx]),
                    'metadata': metadata[idx]
                })
        
        return results
        
    except Exception as e:
        # 오류 시 빈 리스트 반환
        return []

# --------------------------------------------------------------------------------
# 3단계: Hybrid RAG 수렴 로직
# --------------------------------------------------------------------------------
def get_convergence_result_hybrid_rag(
    api_key, aspect, opinion, 
    exact_match_dict, 
    vector_db, metadata, 
    cache=None,
    top_k=50,
    high_similarity_threshold=0.95,
    min_similarity_threshold=0.7,
    model=None  # [NEW] 모델 전달
):
    """
    Hybrid RAG 방식 수렴
    
    1단계: Exact Match (Rule-based)
    2단계: Vector Similarity Search (Top-50, threshold > 0.7)
           - 유사도 0.95 이상이면 즉시 반환
    3단계: AI Few-shot (Dynamic Context: Top-50만 프롬프트에 포함)
    
    Args:
        api_key: Gemini API Key
        aspect: Aspect_Lv2 값
        opinion: Opinion 값
        exact_match_dict: Exact Match용 딕셔너리
        vector_db: 벡터 DB
        metadata: 메타데이터 리스트
        cache: 캐시 딕셔너리 (optional)
        top_k: 검색할 최대 유사 예제 수
        high_similarity_threshold: 즉시 반환할 고유사도 임계값
        min_similarity_threshold: 최소 유사도 임계값
    
    Returns:
        {'Aspect.수렴': ..., 'Opinion.수렴': ...}
    """
    
    # ===== 캐시 체크 =====
    cache_key = (clean_text(aspect), clean_text(opinion))
    if cache and cache_key in cache:
        return cache[cache_key]
    
    # ===== 1단계: Exact Match =====
    if cache_key in exact_match_dict:
        result = exact_match_dict[cache_key]
        if cache is not None:
            cache[cache_key] = result
        return result
    
    # ===== 2단계: Vector Similarity Search =====
    similar_examples = search_similar_examples(
        aspect, opinion, 
        vector_db, metadata, 
        api_key, 
        top_k=top_k, 
        threshold=min_similarity_threshold,
        model=model # [NEW] 모델 전달
    )
    
    # 2-1. 매우 유사한 예제가 있으면 즉시 반환 (0.95 이상)
    if similar_examples and similar_examples[0]['similarity'] >= high_similarity_threshold:
        result = {
            'Aspect.수렴': similar_examples[0]['metadata']['Aspect.수렴'],
            'Opinion.수렴': similar_examples[0]['metadata']['Opinion.수렴']
        }
        if cache is not None:
            cache[cache_key] = result
        return result
    
    # ===== 3단계: AI Few-shot (Dynamic Context) =====
    # Top-50(또는 검색된 예제 전체)만 프롬프트에 포함
    if similar_examples:
        few_shot_examples = "\n".join([
            f"Aspect_Lv2: {ex['metadata']['original_aspect']}, Opinion: {ex['metadata']['original_opinion']} -> "
            f"Aspect.수렴: {ex['metadata']['Aspect.수렴']}, Opinion.수렴: {ex['metadata']['Opinion.수렴']} "
            f"(유사도: {ex['similarity']:.3f})"
            for ex in similar_examples[:top_k]
        ])
    else:
        # 유사 예제가 없으면 원본 반환
        result = {"Aspect.수렴": aspect, "Opinion.수렴": opinion}
        if cache is not None:
            cache[cache_key] = result
        return result
    
    # AI 호출
    try:
        result = _call_gemini_convergence_hybrid(api_key, aspect, opinion, few_shot_examples)
        
        # 결과 검증
        if not result.get('Aspect.수렴') or not result.get('Opinion.수렴'):
            result = {"Aspect.수렴": aspect, "Opinion.수렴": opinion}
        
        if cache is not None:
            cache[cache_key] = result
        
        return result
        
    except Exception as e:
        # 오류 시 원본 반환
        result = {"Aspect.수렴": aspect, "Opinion.수렴": opinion}
        if cache is not None:
            cache[cache_key] = result
        return result

def _call_gemini_convergence_hybrid(api_key, aspect, opinion, few_shot_examples):
    """
    Gemini API를 사용한 수렴 (Hybrid RAG용)
    프롬프트에 동적으로 선택된 유사 예제만 포함
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f'''당신은 VOC 데이터를 정제하는 데이터 분석가입니다.
ㄴ
[SIMILAR EXAMPLES - 입력과 유사한 상위 50개 예제]
{few_shot_examples}

[UNMAPPED KEYWORDS]
Aspect_Lv2: {aspect}
Opinion: {opinion}

[TASK]
위 [SIMILAR EXAMPLES]는 입력과 의미적으로 가장 유사한 예제들입니다 (유사도 순 정렬).
이 패턴을 분석하여 [UNMAPPED KEYWORDS]를 "Aspect.수렴"과 "Opinion.수렴"으로 표준화하세요.

[IMPORTANT RULES]
1. **유사도가 높은 예제일수록 더 참고하세요** (첫 번째 예제가 가장 유사함)
2. 유사한 단어는 반드시 같은 표준어로 수렴시키세요
3. "Aspect.수렴"은 제품의 속성, "Opinion.수렴"은 평가입니다
4. 단순 형용사(좋다, 나쁘다, 비싸다)는 Opinion으로 유지
5. 빈칸 금지. 판단 불가 시 원본 값 사용
6. [강조 표현 제거] Opinion에서 '매우', '너무', '아주', '엄청', '완전', '되게', '갓', '진짜' 등의 강조 부사는 모두 제거하고, 핵심 형용사만 남기세요.
   (예시: "너무 좋음" -> "좋음", "완전 빠름" -> "빠름", "아주 비쌈" -> "비쌈", "진짜 예쁨" -> "예쁨")

JSON 형식으로만 응답:
{{"Aspect.수렴": "...", "Opinion.수렴": "..."}}'''

    response = model.generate_content(prompt)
    json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))
    raise Exception("JSON parsing failed")

# --------------------------------------------------------------------------------
# [수정된 함수] Embedding Matrix 기반 수렴 결과 업데이트 (2단계 적용 로직)
# --------------------------------------------------------------------------------
def apply_embedding_rules_to_result(base_dir):
    """
    Embedding Matrix_VD.csv의 규칙을 AOS_converged_result.csv에 적용합니다.
    (로직: 1. 완전 일치 우선 적용 -> 2. 부분 일치 적용 -> 3. 결과값 재매핑(Chaining)으로 최종 수렴 보장)
    """
    embedding_path = os.path.join(base_dir, "Embedding Matrix_VD.csv")
    result_path = os.path.join(base_dir, "AOS_converged_result.csv")

    if not os.path.exists(embedding_path):
        return False, "Embedding Matrix 파일이 없습니다."
    if not os.path.exists(result_path):
        return False, "수렴 결과 파일(AOS_converged_result.csv)이 없습니다."

    # 내부 헬퍼 함수
    def _clean(text):
        if pd.isna(text) or str(text).strip() == "" or str(text).lower() == 'nan':
            return ""
        return str(text).strip()

    try:
        df_emb = pd.read_csv(embedding_path)
        df_res = pd.read_csv(result_path)

        # 1. 규칙 로딩
        exact_rules = {}   
        aspect_rules = {}  
        opinion_rules = {} 

        for _, row in df_emb.iterrows():
            k_asp = _clean(row.get('Aspect_Lv2', ''))
            k_op = _clean(row.get('Opinion', ''))
            v_asp = _clean(row.get('Aspect.수렴', ''))
            v_op = _clean(row.get('Opinion.수렴', ''))

            if k_asp and k_op:
                exact_rules[(k_asp, k_op)] = (v_asp, v_op)
            elif k_asp and not k_op:
                aspect_rules[k_asp] = (v_asp, v_op)
            elif not k_asp and k_op:
                opinion_rules[k_op] = (v_asp, v_op)

        updated_count = 0
        
        # 2. 데이터 처리
        for idx, row in df_res.iterrows():
            r_asp = _clean(row.get('Aspect_Lv2', ''))
            r_op = _clean(row.get('Opinion', ''))
            
            # (1) 초기값: 현재 저장된 값 (없으면 원본) - LLM 결과 보존
            current_asp = row.get('Aspect.수렴', '')
            current_op = row.get('Opinion.수렴', '')
            if pd.isna(current_asp): current_asp = r_asp
            if pd.isna(current_op): current_op = r_op
            
            cand_asp = str(current_asp)
            cand_op = str(current_op)
            
            # 규칙 적용 여부 (Specific Rule 우선권 확인용)
            exact_applied_and_changed = False

            # (2) 완전 일치 규칙 (Exact Match)
            # - 원본(r_asp, r_op)과 정확히 일치하는 규칙이 있으면 적용
            if (r_asp, r_op) in exact_rules:
                ex_asp, ex_op = exact_rules[(r_asp, r_op)]
                # 값이 '변경'되는 규칙인 경우 적용
                if ex_asp != r_asp or ex_op != r_op:
                    cand_asp = ex_asp
                    cand_op = ex_op
                    exact_applied_and_changed = True
                else:
                    # 값이 같은(Identity) 규칙이면, 부분 규칙이 덮어쓸 수 있도록 패스
                    pass

            # (3) 부분 일치 규칙 (Partial Match on ORIGINAL)
            # - 완전 일치로 이미 '변경'되었다면 부분 규칙 무시 (Specific wins)
            # - 완전 일치가 없거나, 있어도 변경이 없었다면(Identity) 부분 규칙 적용
            if not exact_applied_and_changed:
                if r_asp in aspect_rules:
                    val_asp, val_op = aspect_rules[r_asp]
                    if val_asp: cand_asp = val_asp
                    if val_op: cand_op = val_op
                
                if r_op in opinion_rules:
                    val_asp, val_op = opinion_rules[r_op]
                    if val_asp: cand_asp = val_asp
                    if val_op: cand_op = val_op

            # (4) 연쇄 규칙 적용 (Chaining on CANDIDATE)
            # - 앞선 단계의 결과값(cand_op)이 부분 규칙의 키와 일치하면 다시 적용
            # - 예: "콘텐츠 부족" -> (Exact) -> "부족하다" -> (Chaining) -> "부족"
            curr_op_key = _clean(cand_op)
            if curr_op_key in opinion_rules:
                val_asp, val_op = opinion_rules[curr_op_key]
                if val_asp: cand_asp = val_asp
                if val_op: cand_op = val_op
            
            # (Aspect Chaining은 드물지만 동일하게 처리)
            curr_asp_key = _clean(cand_asp)
            if curr_asp_key in aspect_rules:
                val_asp, val_op = aspect_rules[curr_asp_key]
                if val_asp: cand_asp = val_asp
                if val_op: cand_op = val_op

            # 변경사항 저장
            is_changed = False
            if str(row.get('Aspect.수렴', '')).strip() != cand_asp:
                df_res.at[idx, 'Aspect.수렴'] = cand_asp
                is_changed = True
                
            if str(row.get('Opinion.수렴', '')).strip() != cand_op:
                df_res.at[idx, 'Opinion.수렴'] = cand_op
                is_changed = True
            
            if is_changed:
                updated_count += 1

        if updated_count > 0:
            df_res.to_csv(result_path, index=False, encoding='utf-8-sig')
            if 'df_final' in st.session_state:
                st.session_state['df_final'] = df_res
            return True, f"총 {updated_count}건 업데이트 완료! (완전일치 -> 부분일치 -> 재매핑 적용)"
        else:
            return True, "업데이트할 내역이 없습니다. (모두 최신 상태)"

    except Exception as e:
        return False, f"오류 발생: {str(e)}"

# --------------------------------------------------------------------------------
# [NEW] M 코드 파싱 함수 (Human-In-The-Loop)
# --------------------------------------------------------------------------------
def parse_m_code_to_mapping(m_code_text):
    """
    Power BI M 코드 텍스트에서 Table.ReplaceValue 구문을 찾아
    (Old Value, New Value) 쌍을 추출하고, 적용된 컬럼명에 따라 매핑합니다.
    """
    # 정규표현식 패턴 수정:
    # 마지막 부분의 {"ColumnName"} 까지 캡처하도록 확장
    # 그룹: old(변경전), new(변경후), col(컬럼명)
    pattern = r'Table\.ReplaceValue\(.*,\s*"(?P<old>.*?)",\s*"(?P<new>.*?)",\s*Replacer\.\w+,\s*\{"(?P<col>.*?)"\}\)'
    
    matches = re.finditer(pattern, m_code_text)
    
    extracted_data = []
    for match in matches:
        old_val = match.group("old")
        new_val = match.group("new")
        col_name = match.group("col")
        
        # 기본 빈 딕셔너리 생성
        row_data = {
            "Aspect_Lv2": "", 
            "Aspect.수렴": "",
            "Opinion": "", 
            "Opinion.수렴": ""
        }
        
        # 컬럼명에 따라 데이터 분기 처리
        if "Aspect" in col_name:
            # Aspect 관련 컬럼 수정인 경우
            row_data["Aspect_Lv2"] = old_val
            row_data["Aspect.수렴"] = new_val
            
        elif "Opinion" in col_name:
            # Opinion 관련 컬럼 수정인 경우
            row_data["Opinion"] = old_val
            row_data["Opinion.수렴"] = new_val
            
        else:
            # 그 외 컬럼(예: Sentiment 등)은 일단 Aspect 쪽에 넣거나, 무시할 수 있음
            # 여기서는 Aspect 쪽으로 기본 처리하되 로그를 남길 수도 있음
            row_data["Aspect_Lv2"] = old_val
            row_data["Aspect.수렴"] = new_val

        extracted_data.append(row_data)
        
    return pd.DataFrame(extracted_data)

# ==========================================
# [메인] 탭 구성 (Step 1 ~ Step 7)
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "1. 데이터 수집", 
    "2. 데이터 정제", 
    "3. 키워드 추출", 
    "4. Explode (AOS ID 생성)", 
    "5. 키워드 수렴", 
    "6. 시각화 인사이트", 
    "7. Embedding Matrix 동기화 (HITL)"
])

# ==============================================================================
# [STEP 1] 데이터 수집 (Data Collection)
# ==============================================================================
with tab1:
    st.header("Step 1. 데이터 수집")
    
    crawl_source = st.radio("수집 대상 선택", ["YouTube (API)", "Coupang (Manual HTML)", "Amazon (Manual HTML)", "Best Buy (Manual HTML)", "Samsung.com (Selenium)", "LGE.co.kr (Selenium)"], horizontal=True)
    
    # 1. YouTube
    if crawl_source == "YouTube (API)":
        st.subheader("YouTube 댓글 수집")
        yt_api_key = os.environ.get("YOUTUBE_API_KEY")
        if not yt_api_key:
             yt_api_key = st.text_input("YouTube Data API Key", type="password", key="yt_key_input_tab0")
        
        if 'yt_product_name_val' not in st.session_state:
            st.session_state['yt_product_name_val'] = ""
        product_name = st.text_input("제품명 (Product Name)", value=st.session_state['yt_product_name_val'], key="yt_product_input")
        st.session_state['yt_product_name_val'] = product_name
        
        video_url_or_id = st.text_input("YouTube 동영상 URL 또는 ID", placeholder="https://www.youtube.com/watch?v=... 또는 VideoID")
        
        if st.button("수집 시작 (YouTube)"):
            if not yt_api_key:
                st.error("API Key가 필요합니다.")
            elif not video_url_or_id:
                st.error("동영상 ID가 필요합니다.")
            else:
                vid_id = video_url_or_id
                if "v=" in video_url_or_id:
                    vid_id = video_url_or_id.split("v=")[1].split("&")[0]
                elif "youtu.be/" in video_url_or_id:
                    vid_id = video_url_or_id.split("youtu.be/")[1].split("?")[0]
                elif "shorts/" in video_url_or_id:
                    vid_id = video_url_or_id.split("shorts/")[1].split("?")[0]
                
                crawler = YouTubeCrawler(yt_api_key)
                with st.spinner(f"'{vid_id}' 동영상 정보 및 댓글 수집 중..."):
                    df_yt = crawler.crawl_video(vid_id)
                    if df_yt is not None and not df_yt.empty:
                        df_yt['product_name'] = product_name
                        st.success(f"수집 완료! 총 {len(df_yt)}개 데이터 (댓글 등)")
                        st.dataframe(df_yt.head())
                        save_path = os.path.join(base_dir, "collected_data.csv")
                        save_and_accumulate(df_yt, save_path)
                    else:
                        st.warning("데이터를 찾을 수 없거나 댓글이 없습니다.")

    # 2. Coupang
    elif crawl_source == "Coupang (Manual HTML)":
        st.subheader("Coupang 리뷰 파싱 (HTML 붙여넣기)")
        st.info("쿠팡 상품 페이지에서 리뷰 영역 HTML을 복사하여 아래에 붙여넣으세요.")
        
        if 'cp_product_name_val' not in st.session_state:
            st.session_state['cp_product_name_val'] = ""
        product_name_cp = st.text_input("제품명 (Product Name)", value=st.session_state['cp_product_name_val'], key="cp_product_name")
        st.session_state['cp_product_name_val'] = product_name_cp

        html_input = st.text_area("HTML 소스 코드", height=300)
        
        if st.button("HTML 파싱 및 저장"):
            if html_input:
                parser = CoupangParser()
                df_cp = parser.parse_html(html_input)
                if not df_cp.empty:
                    df_cp['product_name'] = product_name_cp
                    st.success(f"파싱 완료! 총 {len(df_cp)}개 리뷰 추출")
                    st.dataframe(df_cp.head())
                    save_path = os.path.join(base_dir, "collected_data.csv")
                    save_and_accumulate(df_cp, save_path)
                else:
                    st.warning("유효한 리뷰 데이터를 찾지 못했습니다.")

    # 3. Amazon
    elif crawl_source == "Amazon (Manual HTML)":
        st.subheader("Amazon 리뷰 파싱 (HTML 붙여넣기)")
        st.info("아마존 상품 페이지에서 리뷰 영역 HTML을 복사하여 아래에 붙여넣으세요.")
        
        if 'amz_product_name_val' not in st.session_state:
            st.session_state['amz_product_name_val'] = ""
        product_name_amz = st.text_input("제품명 (Product Name)", value=st.session_state['amz_product_name_val'], key="amz_product_name")
        st.session_state['amz_product_name_val'] = product_name_amz

        html_input = st.text_area("HTML 소스 코드 (Amazon)", height=300)
        
        if st.button("Amazon 파싱 및 저장"):
            if html_input:
                parser = AmazonParser()
                df_amz = parser.parse_html(html_input)
                if not df_amz.empty:
                    df_amz['product_name'] = product_name_amz
                    st.success(f"파싱 완료! 총 {len(df_amz)}개 리뷰 추출")
                    st.dataframe(df_amz.head())
                    save_path = os.path.join(base_dir, "collected_data.csv")
                    save_and_accumulate(df_amz, save_path)
                else:
                    st.warning("유효한 Amazon 리뷰 데이터를 찾지 못했습니다.")

    # 3-1. Best Buy
    elif crawl_source == "Best Buy (Manual HTML)":
        st.subheader("Best Buy Review Parsing (Manual HTML)")
        st.info("Copy the review section HTML from Best Buy product page and paste it below.")
        
        if 'bb_product_name_val' not in st.session_state:
            st.session_state['bb_product_name_val'] = ""
        product_name_bb = st.text_input("Product Name", value=st.session_state['bb_product_name_val'], key="bb_product_name")
        st.session_state['bb_product_name_val'] = product_name_bb

        html_input = st.text_area("HTML Source Code (Best Buy)", height=300)
        
        if st.button("Parse Best Buy & Save"):
            if html_input:
                parser = BestBuyParser()
                df_bb = parser.parse_html(html_input)
                if not df_bb.empty:
                    df_bb['product_name'] = product_name_bb
                    st.success(f"Parsing Complete! Extracted {len(df_bb)} reviews.")
                    st.dataframe(df_bb.head())
                    save_path = os.path.join(base_dir, "collected_data.csv")
                    save_and_accumulate(df_bb, save_path)
                else:
                    st.warning("No valid Best Buy reviews found.")

    # 4. Samsung.com
    elif crawl_source == "Samsung.com (Selenium)":
        st.subheader("Samsung.com 리뷰 크롤링")
        st.warning("※ 실행 시 Chrome 브라우저가 자동으로 실행됩니다.")
        
        if 'ss_product_name_val' not in st.session_state:
            st.session_state['ss_product_name_val'] = ""
        product_name_ss = st.text_input("제품명 (Product Name)", value=st.session_state['ss_product_name_val'], key="ss_product_name")
        st.session_state['ss_product_name_val'] = product_name_ss
        
        target_url = st.text_input("대상 URL (Samsung.com)", placeholder="https://www.samsung.com/sec/...")
        max_pages = st.number_input("수집할 최대 페이지 수", min_value=1, value=13)
        
        if st.button("수집 시작 (Samsung)"):
            if not target_url:
                st.error("URL을 입력해주세요.")
            else:
                crawler = SamsungCrawler()
                try:
                    with st.spinner(f"Samsung.com 크롤링 중... (최대 {max_pages} 페이지)"):
                        df_ss = crawler.crawl_reviews(target_url, max_pages=max_pages)
                except Exception as e:
                    st.error(f"크롤링 중 오류 발생: {e}")
                    df_ss = None
                finally:
                    crawler.close()
                
                if df_ss is not None and not df_ss.empty:
                    df_ss['product_name'] = product_name_ss
                    st.success(f"수집 완료! 총 {len(df_ss)}개 리뷰")
                    st.dataframe(df_ss.head())
                    save_path = os.path.join(base_dir, "collected_data.csv")
                    save_and_accumulate(df_ss, save_path)
                else:
                    st.warning("수집된 데이터가 없습니다.")

    # 5. LGE.co.kr
    elif crawl_source == "LGE.co.kr (Selenium)":
        st.subheader("LGE.co.kr 리뷰 크롤링")
        st.warning("※ 실행 시 Chrome 브라우저가 자동으로 실행됩니다.")
        
        if 'lg_product_name_val' not in st.session_state:
            st.session_state['lg_product_name_val'] = ""
        product_name_lg = st.text_input("제품명 (Product Name)", value=st.session_state['lg_product_name_val'], key="lg_product_name")
        st.session_state['lg_product_name_val'] = product_name_lg
        
        target_url = st.text_input("대상 URL (LGE.co.kr)", placeholder="https://www.lge.co.kr/...")
        max_more_clicks = st.number_input("더보기 버튼 클릭 횟수", min_value=1, value=5)
        
        if st.button("수집 시작 (LG)"):
            if not target_url:
                st.error("URL을 입력해주세요.")
            else:
                crawler = LGCrawler()
                try:
                    with st.spinner(f"LGE.co.kr 크롤링 중... (더보기 {max_more_clicks}회)"):
                        df_lg = crawler.crawl_reviews(target_url, max_more_clicks=max_more_clicks)
                except Exception as e:
                    st.error(f"크롤링 중 오류 발생: {e}")
                    df_lg = None
                finally:
                    crawler.close()
                
                if df_lg is not None and not df_lg.empty:
                    df_lg['product_name'] = product_name_lg
                    st.success(f"수집 완료! 총 {len(df_lg)}개 리뷰")
                    st.dataframe(df_lg.head())
                    save_path = os.path.join(base_dir, "collected_data.csv")
                    save_and_accumulate(df_lg, save_path)
                else:
                    st.warning("수집된 데이터가 없습니다.")

# ... (나머지 탭들 유지) ...

# ==============================================================================
# [STEP 2] 데이터 정제 (Data Cleaning)
# ==============================================================================
with tab2:
    st.header("Step 2. 데이터 정제")
    uploaded_file = st.file_uploader("CSV 파일 업로드 (예: dm crawl_list_info_VD.csv)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"총 {len(df)}건의 데이터가 로드되었습니다.")
    elif 'df_collected' in st.session_state:
        st.info("0단계에서 수집된 데이터를 불러옵니다.")
        df = st.session_state['df_collected']
        st.write(f"수집된 데이터: {len(df)}건")
    else:
        df = None

    if df is not None:
        st.dataframe(df.head())
        target_col = st.selectbox("분석할 리뷰 컬럼 선택", df.columns, index=list(df.columns).index("contents") if "contents" in df.columns else 0)
        st.session_state['target_review_col'] = target_col
        
        if st.button("정제 시작"):
            irrelevant_keywords = ["잘 봤습니다", "좋은 정보", "퍼가요", "쪽지 주세요", "구독", "수고하셨습니다", "형님", "누님", "광고","김구", "협찬", "비밀댓글", "삭제된 댓글","구독과 좋아요"]
            
            valid_rows = []
            removed_count = 0
            progress_bar = st.progress(0)
            df.reset_index(drop=True, inplace=True)
            
            for i, row in df.iterrows():
                content = str(row[target_col]).replace(',', '') 
                is_valid, reason = is_valid_sentence(content, irrelevant_keywords)
                if is_valid:
                    row[target_col] = content
                    valid_rows.append(row)
                else:
                    removed_count += 1
                if i % 100 == 0:
                    progress_bar.progress((i + 1) / len(df))
            
            progress_bar.progress(1.0)
            df_cleaned = pd.DataFrame(valid_rows)
            st.success(f"정제 완료! (제거됨: {removed_count}건, 남음: {len(df_cleaned)}건)")
            st.session_state['df_cleaned'] = df_cleaned
            
            save_path = os.path.join(base_dir, "cleaned_data.csv")
            df_cleaned.to_csv(save_path, index=False, encoding='utf-8-sig')
            st.toast(f"💾 결과가 자동으로 저장되었습니다: {save_path}")
            
            csv = df_cleaned.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button("정제된 CSV 다운로드", csv, "cleaned_data.csv", "text/csv")

# ==============================================================================
# [STEP 3] 키워드 추출 (AI Keyword Extraction)
# ==============================================================================
with tab3:
    st.header("Step 3. 키워드 추출")
    
    if 'df_cleaned' in st.session_state:
        df_target = st.session_state['df_cleaned']
    else:
        st.info("1단계에서 정제를 먼저 수행하거나, 정제된 파일을 업로드하세요.")
        uploaded_cleaned = st.file_uploader("정제된 CSV 업로드", type=["csv"], key="upload_cleaned")
        if uploaded_cleaned:
            df_target = pd.read_csv(uploaded_cleaned)
            st.session_state['df_cleaned'] = df_target
        else:
            df_target = None

    if df_target is not None:
        st.write(f"분석 대상: {len(df_target)}건")
        
        # [NEW] 컬럼 선택 추가 (Tab 1에서 선택한 값 연동)
        default_col_idx = 0
        if 'target_review_col' in st.session_state and st.session_state['target_review_col'] in df_target.columns:
             default_col_idx = list(df_target.columns).index(st.session_state['target_review_col'])
        elif "contents" in df_target.columns:
             default_col_idx = list(df_target.columns).index("contents")
        review_col = st.selectbox("분석할 리뷰 컬럼 선택 (추출)", df_target.columns, index=default_col_idx, key="extract_col_select")
        
        if 'level1_recommended' not in st.session_state:
            st.session_state['level1_recommended'] = "화질.디스플레이, 디자인, 소프트웨어/UX, 가격, 내구성.안전, 음질.사운드, 연결성.호환성, 설치.설정, 게임, 성능.기능, 에너지.환경, 이동성.편의성, 기타"

        col_cat1, col_cat2 = st.columns([4, 1])
        with col_cat1:
            level1_cats = st.text_area("Level 1 카테고리 (콤마로 구분)", 
                                       value=st.session_state['level1_recommended'],
                                       help="추출 시 사용할 상위 범주입니다. 직접 수정하거나 AI 추천을 받으세요.")
        with col_cat2:
            st.write("") # 간격 맞추기용
            st.write("")
            if st.button("🪄 AI 추천", help="수집된 데이터를 분석하여 최적의 카테고리를 추천합니다."):
                if not api_key:
                    st.error("API 키를 먼저 입력해주세요.")
                elif 'df_collected' not in st.session_state and 'df_cleaned' not in st.session_state:
                    st.error("수집되거나 정제된 데이터가 없습니다.")
                else:
                    with st.spinner("데이터 분석 중..."):
                        # 데이터 샘플링 (최대 100건)
                        df_sample = st.session_state.get('df_cleaned', st.session_state.get('df_collected'))
                        sample_texts = df_sample[review_col].dropna().sample(min(100, len(df_sample))).tolist()
                        combined_samples = "\n".join([f"- {t[:200]}" for t in sample_texts])
                        
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-2.0-flash') # 추천용은 flash 모델로 충분
                        
                        prompt = f"""
                        당신은 제품 리뷰 분석 전문가입니다. 다음은 고객들의 리뷰 샘플들입니다.
                        이 리뷰들을 분석하여, 전체 데이터를 분류하기에 가장 적합한 10~15개의 상위 카테고리(Aspect Level 1)를 뽑아주세요.
                        
                        [리뷰 샘플]
                        {combined_samples}
                        
                        [출력 형식]
                        - 카테고리 이름만 콤마(,)로 구분하여 한 줄로 출력하세요.
                        - 예: 화질,음질,디자인,가격,성능,기능,배송,기타
                        - 가급적 의미가 명확하고 중복되지 않게 구성하세요.
                        - 마지막에는 항상 '기타'를 포함하세요.
                        """
                        
                        try:
                            response = model.generate_content(prompt)
                            recommended = response.text.strip().replace('\n', '')
                            # 불필요한 공백 및 따옴표 제거
                            recommended = re.sub(r'["\'`\-\*]', '', recommended)
                            st.session_state['level1_recommended'] = recommended
                            st.rerun()
                        except Exception as e:
                            st.error(f"추천 생성 실패: {e}")

        level1_list = [c.strip() for c in level1_cats.split(',')]
        
        if st.button("추출 시작 (Gemini API)"):
            if not api_key:
                st.error("API 키가 설정되지 않았습니다.")
            else:
                extractor = KeywordExtractor(api_key, level1_list)
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # review_col은 위에서 이미 선택됨
                # review_col = "contents" if "contents" in df_target.columns else df_target.columns[0]
                save_path = os.path.join(base_dir, "voc_results.csv")
                start_idx = 0
                existing_results = []
                
                if os.path.exists(save_path):
                    try:
                        df_existing = pd.read_csv(save_path)
                        start_idx = len(df_existing)
                        existing_results = df_existing.to_dict('records')
                        st.info(f"📂 이전 작업 발견: {start_idx}건부터 이어서 시작합니다.")
                    except:
                        start_idx = 0
                
                df_to_process = df_target.iloc[start_idx:]
                
                if len(df_to_process) == 0:
                    st.success("모든 데이터가 이미 분석되었습니다!")
                    st.dataframe(pd.read_csv(save_path).head())
                else:
                    status_container = st.empty()
                    result_container = st.empty()
                    results = existing_results
                    total_process = len(df_target)
                    file_lock = threading.Lock() 
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {}
                        for loop_idx, (index, row) in enumerate(df_to_process.iterrows()):
                            text = str(row[review_col])
                            future = executor.submit(extractor.extract_keywords_with_retry, text)
                            futures[future] = (index, row, loop_idx)
                        
                        total_process = len(futures)
                        completed_count = 0
                        
                        status_container = st.empty()
                        result_container = st.empty()
                        
                        for future in concurrent.futures.as_completed(futures):
                            index, row, loop_idx = futures[future]
                            res = future.result()
                            keywords = res.get("keywords", [])
                            
                            completed_count += 1
                            current_progress = start_idx + completed_count
                            
                            with result_container.container():
                                st.write(f"✅ **[{current_progress}/{start_idx + total_process}] 완료**: {str(row[review_col])[:30]}...")
                            
                            lv1, lv2, op, sent = preprocess_keywords(keywords, level1_list)
                            
                            row_dict = row.to_dict()
                            row_dict.update({
                                "Aspect_Lv1s": lv1, "Aspect_Lv2s": lv2, 
                                "Opinions": op, "Sentiments": sent
                            })
                            
                            with file_lock:
                                existing_results.append(row_dict)
                                df_current_row = pd.DataFrame([row_dict])
                                header = not os.path.exists(save_path)
                                df_current_row.to_csv(save_path, mode='a', index=False, header=header, encoding='utf-8-sig')
                            
                            if completed_count % 1 == 0:
                                progress_val = min(completed_count / total_process, 1.0)
                                progress_bar.progress(progress_val)
                                status_text.text(f"진행률: {int(progress_val*100)}% (동시 처리 중...)")
                                status_container.info(f"🚀 분석 속도 향상 [병렬]... [{completed_count} / {total_process}]")

                    progress_bar.progress(1.0)
                    status_text.text("완료!")
                    status_container.success("모든 분석이 완료되었습니다.")
                    
                    df_extracted = pd.read_csv(save_path)
                    st.session_state['df_extracted'] = df_extracted
                    st.dataframe(df_extracted.head())
                    csv = df_extracted.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                    st.download_button("추출 결과 다운로드 (voc_results.csv)", csv, "voc_results.csv", "text/csv")

# ==============================================================================
# [STEP 4] Explode (AOS ID 생성)
# ==============================================================================
with tab4:
    st.header("Step 4. Explode (AOS ID 생성)")
    
    if 'df_extracted' in st.session_state:
        df_ex = st.session_state['df_extracted']
    else:
        st.info("2단계 결과를 업로드하세요.")
        uploaded_ex = st.file_uploader("추출 결과 CSV 업로드", type=["csv"], key="upload_ex")
        if uploaded_ex:
            df_ex = pd.read_csv(uploaded_ex)
            st.session_state['df_extracted'] = df_ex
        else:
            df_ex = None
            
    if df_ex is not None:
        if st.button("전처리 실행"):
            exploded_rows = []
            aos_id = 1
            
            for _, row in df_ex.iterrows():
                lv1s = str(row.get('Aspect_Lv1s', '')).split('|')
                lv2s = str(row.get('Aspect_Lv2s', '')).split('|')
                ops = str(row.get('Opinions', '')).split('|')
                sents = str(row.get('Sentiments', '')).split('|')
                
                max_len = max(len(lv1s), len(lv2s), len(ops), len(sents))
                
                def get_val(arr, idx):
                    if idx < len(arr): return arr[idx].strip()
                    if len(arr) > 0: return arr[-1].strip()
                    return ""

                for i in range(max_len):
                    op_val = get_val(ops, i)
                    if not op_val: continue
                    op_clean = str(op_val).strip()
                    if op_clean == "" or op_clean == "-" or op_clean.lower() == "nan": continue
                    
                    new_row = row.to_dict()
                    for k in ['Aspect_Lv1s', 'Aspect_Lv2s', 'Opinions', 'Sentiments']:
                        if k in new_row: del new_row[k]
                        
                    new_row['Aspect_Lv1'] = get_val(lv1s, i)
                    new_row['Aspect_Lv2'] = get_val(lv2s, i)
                    new_row['Opinion'] = op_val
                    new_row['Sentiment'] = get_val(sents, i)
                    new_row['AOS_ID'] = aos_id
                    aos_id += 1
                    
                    exploded_rows.append(new_row)
            
            df_exploded = pd.DataFrame(exploded_rows)
            st.session_state['df_exploded'] = df_exploded
            st.success(f"전처리 완료! 총 {len(df_exploded)}개의 개별 의견이 생성되었습니다.")
            st.dataframe(df_exploded.head())
            
            save_path = os.path.join(base_dir, "result_python_final.csv")
            df_exploded.to_csv(save_path, index=False, encoding='utf-8-sig')
            st.toast(f"💾 결과가 자동으로 저장되었습니다: {save_path}")
            
            csv = df_exploded.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button("전처리 결과 다운로드 (result_python_final.csv)", csv, "result_python_final.csv", "text/csv")

# ==============================================================================
# [STEP 5] 키워드 수렴 (Keyword Convergence)
# ==============================================================================
with tab5:
    st.header("Step 5. 키워드 수렴")
    st.info("🚀 **NEW**: 벡터 유사도 검색을 통한 Hybrid RAG 방식 적용! (속도 5~10배 향상)")
    
    if 'df_exploded' in st.session_state:
        df_conv_input = st.session_state['df_exploded']
    else:
        st.info("3단계 결과를 업로드하세요.")
        uploaded_conv = st.file_uploader("전처리 결과 CSV 업로드", type=["csv"], key="upload_conv")
        if uploaded_conv:
            df_conv_input = pd.read_csv(uploaded_conv)
            st.session_state['df_exploded'] = df_conv_input
        else:
            df_conv_input = None
            
    if 'df_final' in st.session_state:
        st.info("✅ 이전에 작업한 수렴 결과가 로드되었습니다.")
        st.dataframe(st.session_state['df_final'].head())
        csv = st.session_state['df_final'].to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button("최종 결과 다운로드 (AOS_converged_result.csv)", csv, "AOS_converged_result.csv", "text/csv", key="download_final_loaded")
        st.divider()
        st.write("다시 실행하시겠습니까?")
    
    # ===== Embedding Matrix 업로드 및 벡터 DB 구축 =====
    st.subheader("📚 Embedding Matrix 설정")
    uploaded_embedding = st.file_uploader("Embedding Matrix CSV 업로드 (예: Embedding Matrix_VD.csv)", type=["csv"], key="upload_embedding")
    
    exact_match_dict = {}
    vector_db = None
    metadata = None
    
    if uploaded_embedding:
        df_embedding = pd.read_csv(uploaded_embedding)
        st.success(f"✅ Embedding Matrix 로드 완료 ({len(df_embedding)}건)")
        
        # 1. Exact Match 딕셔너리 구축
        for _, row in df_embedding.iterrows():
            k = (clean_text(row.get('Aspect_Lv2', '')), clean_text(row.get('Opinion', '')))
            v = {
                'Aspect.수렴': clean_text(row.get('Aspect.수렴', '')),
                'Opinion.수렴': clean_text(row.get('Opinion.수렴', ''))
            }
            exact_match_dict[k] = v
        
        # 2. 벡터 DB 구축 (Gemini Embedding API)
        # 2. 벡터 DB 구축 (로컬 모델 + 파일 캐싱)
        vector_db_path = os.path.join(base_dir, "vector_db.pkl")
        
        # 2. 벡터 DB 구축 (로컬 모델 + 파일 캐싱)
        vector_db_path = os.path.join(base_dir, "vector_db.pkl")
        
        # 해시 생성 (안정성을 위해 데이터 내용 기반으로 변경: CSV 변환 후 해싱)
        # tobytes()는 메모리 구조에 따라 달라질 수 있어 불안정함
        try:
            content_bytes = df_embedding.to_csv(index=False).encode('utf-8')
            current_hash = hashlib.md5(content_bytes).hexdigest()
        except:
            current_hash = hashlib.md5(str(df_embedding).encode()).hexdigest()
        
        # 세션에 없거나 해시가 다르면 로드/구축 시도
        if 'vector_db' not in st.session_state or st.session_state.get('vector_db_hash') != current_hash:
            
            loaded_from_file = False
            # 1. 파일에서 로드 시도
            if os.path.exists(vector_db_path):
                try:
                    with open(vector_db_path, 'rb') as f:
                        saved_data = pickle.load(f)
                    
                    saved_hash = saved_data.get('hash')
                    
                    # 데이터 정합성 체크 (해시 비교)
                    if saved_hash == current_hash:
                        vector_db = saved_data['vector_db']
                        metadata = saved_data['metadata']
                        
                        st.session_state['vector_db'] = vector_db
                        st.session_state['vector_metadata'] = metadata
                        st.session_state['vector_db_hash'] = current_hash
                        
                        st.success(f"📂 파일에서 벡터 DB 로드 완료! ({len(vector_db)}개 벡터)")
                        loaded_from_file = True
                    else:
                        st.info(f"🔄 데이터 변경 감지 (DB 재구축 필요)")
                        # 디버깅용 (필요 시 주석 해제)
                        # st.write(f"Saved: {saved_hash[:8]}... vs Current: {current_hash[:8]}...")
                except Exception as e:
                    st.warning(f"기존 벡터 DB 파일 로드 실패 (재구축합니다): {e}")

            # 2. 파일이 없거나 해시가 다르면 새로 구축
            if not loaded_from_file:
                st.warning("🔄 벡터 DB를 구축 중입니다... (최초 1회만 수행, 약 1~2분 소요)")
                
                vector_progress = st.progress(0)
                vector_status = st.empty()
                
                def update_vector_progress(progress):
                    vector_progress.progress(progress)
                    vector_status.text(f"벡터화 진행: {int(progress*100)}%")
                
                try:
                    vector_db, metadata = build_vector_index(df_embedding, api_key, progress_callback=update_vector_progress)
                    
                    # 세션 상태에 저장 (캐싱)
                    st.session_state['vector_db'] = vector_db
                    st.session_state['vector_metadata'] = metadata
                    st.session_state['vector_db_hash'] = current_hash
                    
                    vector_progress.progress(1.0)
                    vector_status.empty()
                    st.success(f"✅ 벡터 DB 구축 완료! ({len(vector_db)}개 벡터 생성)")
                    
                    # 파일로 저장 (영구 캐싱)
                    try:
                        with open(vector_db_path, 'wb') as f:
                            pickle.dump({
                                'hash': current_hash,
                                'vector_db': vector_db,
                                'metadata': metadata
                            }, f, protocol=pickle.HIGHEST_PROTOCOL)
                        st.toast(f"💾 벡터 DB가 파일로 저장되었습니다: vector_db.pkl")
                    except Exception as e:
                        st.error(f"벡터 DB 파일 저장 실패: {e}")
                    
                except Exception as e:
                    st.error(f"벡터 DB 구축 실패: {e}")
                    # 설치 안내 추가
                    if "sentence_transformers" in str(e) or "sentence-transformers" in str(e):
                        st.error("🚨 필수 라이브러리가 없습니다. 터미널에서 다음을 실행하세요:")
                        st.code("pip install sentence-transformers")
                    vector_db = None
                    metadata = None
        else:
            # 세션 캐시 사용
            vector_db = st.session_state['vector_db']
            metadata = st.session_state['vector_metadata']
            st.success(f"✅ 캐싱된 벡터 DB 사용 ({len(vector_db)}개 벡터)")
    else:
        st.warning("⚠️ Embedding Matrix가 없으면 Hybrid RAG를 사용할 수 없습니다.")

    # ===== 수렴 실행 (Hybrid RAG) =====
    if df_conv_input is not None and vector_db is not None:
        st.markdown("---")
        st.subheader("🚀 Hybrid RAG 수렴 실행")
        
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("유사 예제 검색 수 (Top-K)", min_value=10, max_value=100, value=50, step=10,
                             help="각 입력마다 벡터 DB에서 가장 유사한 예제를 몇 개 검색할지 설정")
        with col2:
            similarity_threshold = st.slider("최소 유사도 임계값", min_value=0.5, max_value=0.95, value=0.7, step=0.05,
                                            help="이 값보다 낮은 유사도는 무시됩니다")
        
        if st.button("🔥 Hybrid RAG 수렴 시작"):
            if not api_key:
                st.error("API 키가 필요합니다.")
            else:
                # 캐시 초기화
                if 'convergence_cache' not in st.session_state:
                    st.session_state['convergence_cache'] = {}
                convergence_cache = st.session_state['convergence_cache']
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                save_path = os.path.join(base_dir, "AOS_converged_result.csv")
                start_idx = 0
                existing_rows = []
                
                if os.path.exists(save_path):
                    try:
                        df_existing = pd.read_csv(save_path)
                        start_idx = len(df_existing)
                        existing_rows = df_existing.to_dict('records')
                        st.info(f"📂 이전 작업 발견: {start_idx}건부터 이어서 시작합니다.")
                    except:
                        start_idx = 0
                
                # [NEW] 모델 사전 로드 (중복 로드 방지)
                try:
                    from sentence_transformers import SentenceTransformer
                    st.info("🧠 로컬 AI 모델을 로드 중입니다... (최초 1회, 잠시만 기다려주세요)")
                    # device='cpu'로 명시하여 스레드 충돌/Meta Tensor 에러 방지
                    local_model = SentenceTransformer('BM-K/KoSimCSE-roberta', device='cpu')
                except Exception as e:
                    st.error(f"모델 로드 실패: {e}")
                    st.stop()
                
                df_to_process = df_conv_input.iloc[start_idx:]
                
                if len(df_to_process) == 0:
                    st.success("모든 데이터가 이미 수렴되었습니다!")
                    st.dataframe(pd.read_csv(save_path).head())
                else:
                    status_container = st.empty()
                    result_container = st.empty()
                    total_process = len(df_to_process)
                    file_lock = threading.Lock()
                    
                    # 병렬 처리
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {}
                        for loop_idx, (index, row) in enumerate(df_to_process.iterrows()):
                            aspect = str(row.get('Aspect_Lv2', ''))
                            opinion = str(row.get('Opinion', ''))
                            
                            # Hybrid RAG 함수 호출
                            future = executor.submit(
                                get_convergence_result_hybrid_rag,
                                api_key, aspect, opinion,
                                exact_match_dict,
                                vector_db, metadata,
                                convergence_cache,
                                top_k=top_k,
                                high_similarity_threshold=0.95,
                                min_similarity_threshold=similarity_threshold,
                                model=local_model
                            )
                            futures[future] = (index, row, loop_idx)
                        
                        completed_count = 0
                        
                        for future in concurrent.futures.as_completed(futures):
                            index, row, loop_idx = futures[future]
                            res = future.result()
                            
                            row['Aspect.수렴'] = res.get('Aspect.수렴', row.get('Aspect_Lv2', ''))
                            row['Opinion.수렴'] = res.get('Opinion.수렴', row.get('Opinion', ''))
                            
                            if not row['Aspect.수렴']: row['Aspect.수렴'] = row.get('Aspect_Lv2', '')
                            if not row['Opinion.수렴']: row['Opinion.수렴'] = row.get('Opinion', '')

                            completed_count += 1
                            current_progress = start_idx + completed_count

                            with result_container.container():
                                st.write(f"✅ **[{current_progress}/{start_idx + total_process}] 수렴**: {row.get('Aspect_Lv2', '')} → {row['Aspect.수렴']}")
                            
                            with file_lock:
                                existing_rows.append(row)
                                df_current_row = pd.DataFrame([row])
                                header = not os.path.exists(save_path)
                                df_current_row.to_csv(save_path, mode='a', index=False, header=header, encoding='utf-8-sig')
                            
                            if completed_count % 1 == 0:
                                progress_val = min(completed_count / total_process, 1.0)
                                progress_bar.progress(progress_val)
                                status_text.text(f"진행률: {int(progress_val*100)}% (Hybrid RAG 병렬 처리)")
                                status_container.info(f"🚀 벡터 검색 기반 수렴 중... [{completed_count} / {total_process}]")
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ 완료!")
                    status_container.success(f"🎉 Hybrid RAG 수렴 완료! (캐시 히트율: {len(convergence_cache)}/{total_process})")
                    
                    df_final = pd.read_csv(save_path)
                    st.session_state['df_final'] = df_final
                    st.dataframe(df_final.head())
                    csv = df_final.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                    st.download_button("최종 결과 다운로드 (AOS_converged_result.csv)", csv, "AOS_converged_result.csv", "text/csv")
    
    elif df_conv_input is not None and vector_db is None:
        st.warning("⚠️ Embedding Matrix를 먼저 업로드하고 벡터 DB를 구축해주세요!")

# ==============================================================================
# [STEP 6] 시각화 인사이트 (Visualization)
# ==============================================================================
with tab6:
    st.header("Step 6. 시각화 인사이트")
    st.info("Power BI 대시보드 URL을 입력하면 화면에 바로 표시됩니다.")
    
    saved_url = config.get("power_bi_url", "")
    power_bi_url = st.text_input("Power BI 게시(Publish) URL 입력", value=saved_url, placeholder="https://app.powerbi.com/view?r=...")
    
    if power_bi_url != saved_url:
        config["power_bi_url"] = power_bi_url
        save_config(config)
        st.toast("✅ Power BI URL이 저장되었습니다.")
        webbrowser.open_new_tab(power_bi_url)
        time.sleep(1) 
        st.rerun()

    if power_bi_url:
        st.write("---")
        st.info("아래 버튼을 클릭하여 대시보드를 확인하세요.")
        st.markdown(f"""
            <a href="{power_bi_url}" target="_blank" style="text-decoration: none;">
                <div style="
                    background-color: #f0f2f6;
                    border: 1px solid #d6d6d8;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    font-weight: bold;
                    color: #31333F;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    📊 Power BI 대시보드 새 창에서 열기 ↗️
                </div>
            </a>
        """, unsafe_allow_html=True)
    else:
        st.warning("Power BI URL을 입력해주세요.")

# ==============================================================================
# [STEP 7] Embedding Matrix 동기화 (HITL)
# ==============================================================================
with tab7:
    st.header("Step 7. Embedding Matrix 동기화 (HITL)")
    st.markdown("""
    **Power BI에서 '값 바꾸기'를 수행한 후, '고급 편집기'의 M 코드를 여기에 붙여넣으세요.**
    이 기능은 사용자가 시각적으로 발견하고 수정한 오류(오타, 동의어 등)를 시스템 자산(Embedding Matrix)으로 축적합니다.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 1️⃣ M 코드 입력")
        m_code_input = st.text_area(
            "Power BI 고급 편집기 코드 붙여넣기", 
            height=300, 
            placeholder='= Table.ReplaceValue(#"Changed Type", "가성비", "가격", Replacer.ReplaceText, {"Aspect_Lv2"}) ...'
        )
    
    with col2:
        st.markdown("### 2️⃣ 추출 및 저장")
        if m_code_input:
            df_parsed = parse_m_code_to_mapping(m_code_input)
            
            if not df_parsed.empty:
                st.success(f"✅ 총 {len(df_parsed)}건의 수정 내역을 추출했습니다.")
                st.dataframe(df_parsed, use_container_width=True)
                
                # 빈 컬럼 채우기 (포맷 맞추기)
                if 'Opinion' not in df_parsed.columns:
                    df_parsed['Opinion'] = "" 
                if 'Opinion.수렴' not in df_parsed.columns:
                    df_parsed['Opinion.수렴'] = ""
                
                st.divider()
                st.write("▼ 아래 버튼을 누르면 Embedding Matrix 파일에 이 규칙들이 추가됩니다.")
                
                # 저장 로직
                # 파일명은 사용자가 Tab 4에서 사용하는 'Embedding Matrix_VD.csv'라고 가정하거나 기본 경로 사용
                mapping_file_name = "Embedding Matrix_VD.csv" # 기본값
                mapping_file_path = os.path.join(base_dir, mapping_file_name)
                
                if st.button(f"'{mapping_file_name}'에 추가 및 저장"):
                    # 기존 파일 로드 또는 생성
                    if os.path.exists(mapping_file_path):
                        try:
                            df_existing_map = pd.read_csv(mapping_file_path)
                        except:
                             df_existing_map = pd.DataFrame(columns=['Aspect_Lv2', 'Opinion', 'Aspect.수렴', 'Opinion.수렴'])
                    else:
                        df_existing_map = pd.DataFrame(columns=['Aspect_Lv2', 'Opinion', 'Aspect.수렴', 'Opinion.수렴'])
                    
                    # 병합
                    df_final_map = pd.concat([df_existing_map, df_parsed], ignore_index=True)
                    
                    # 중복 제거 (Aspect_Lv2가 같으면 최신 규칙으로 덮어쓰기 위해 keep='last')
                    df_final_map.drop_duplicates(subset=['Aspect_Lv2', 'Opinion'], keep='last', inplace=True)
                    
                    # 저장
                    try:
                        df_final_map.to_csv(mapping_file_path, index=False, encoding='utf-8-sig')
                        st.toast(f"✅ 저장 완료! 총 {len(df_final_map)}개의 매핑 규칙이 저장되었습니다.")
                        st.success(f"파일이 업데이트되었습니다: {mapping_file_path}")
                        st.dataframe(df_final_map.tail(5))
                    except Exception as e:
                        st.error(f"저장 실패: {e}")
                    
            else:
                st.warning("입력된 코드에서 'Table.ReplaceValue' 패턴을 찾지 못했습니다. 코드를 확인해주세요.")
                st.info("예시: = Table.ReplaceValue(#\"이전단계\", \"오타\", \"정상값\", Replacer.ReplaceText, {\"Aspect_Lv2\"})")

    st.markdown("---")
    st.markdown("### 3️⃣ 수렴 결과에 규칙 적용")
    st.markdown("""
    위에서 저장한 **Embedding Matrix**의 최신 규칙을 기존 **AOS_converged_result.csv** 파일에 즉시 적용합니다.
    AI 수렴을 다시 돌리지 않고도 변경된 규칙을 빠르게 반영할 수 있습니다.
    """)
    
    if st.button("Embedding Matrix 규칙을 수렴 결과에 적용하기"):
        with st.spinner("규칙 적용 중..."):
            success, msg = apply_embedding_rules_to_result(base_dir)
            if success:
                st.success(f"✅ 작업 완료: {msg}")
            else:
                st.error(f"❌ 실패: {msg}")