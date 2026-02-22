# -*- coding: utf-8 -*-
import os
import re
import json
import pandas as pd
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from bs4 import BeautifulSoup
import time
import json

def format_metadata(meta_dict):
    """딕셔너리를 입력받아 JSON 형태의 데이터를 'key: value | key: value' 문자열로 변환합니다."""
    if not meta_dict:
        return ""
    items = []
    for k, v in meta_dict.items():
        if v is not None and str(v).strip() != "":
            items.append(f"{k}: {v}")
    return " | ".join(items)

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

class YouTubeCrawler:
    def __init__(self, api_key):
        self.api_key = api_key
        self.youtube = self._get_youtube_service()

    def _get_youtube_service(self):
        try:
            return build("youtube", "v3", developerKey=self.api_key)
        except HttpError as e:
            print(f"API 서비스 생성 중 오류 발생: {e}")
            return None

    def get_video_details(self, video_ids):
        """동영상 ID 목록을 받아 상세 정보(제목, 좋아요, 조회수)를 반환합니다."""
        if not self.youtube: return []
        
        video_details = []
        # API는 한 번에 최대 50개의 ID만 처리할 수 있습니다.
        for i in range(0, len(video_ids), 50):
            batch_ids = video_ids[i:i+50]
            try:
                request = self.youtube.videos().list(
                    part="snippet,statistics",
                    id=",".join(batch_ids)
                )
                response = request.execute()
                
                for item in response.get("items", []):
                    snippet = item.get("snippet", {})
                    statistics = item.get("statistics", {})
                    
                    video_details.append({
                        "id": item["id"],
                        "title": snippet.get("title"),
                        "publishedAt": snippet.get("publishedAt"),
                        "viewCount": int(statistics.get("viewCount", 0)),
                        "likeCount": int(statistics.get("likeCount", 0)),
                        "commentCount": int(statistics.get("commentCount", 0))
                    })
            except HttpError as e:
                print(f"동영상 상세 정보 조회 중 오류 발생 (ID: {batch_ids}): {e}")
                
        return video_details

    def get_comment_threads(self, video_id, max_results=100):
        """동영상의 최상위 댓글과 내용을 가져옵니다. (Pagination 적용)"""
        if not self.youtube: return []
        
        comments_data = []
        next_page_token = None
        
        while True:
            try:
                request = self.youtube.commentThreads().list(
                    part="snippet,replies",
                    videoId=video_id,
                    maxResults=100, # 페이지당 최대 100개
                    textFormat="plainText",
                    pageToken=next_page_token
                )
                response = request.execute()
                
                for item in response.get("items", []):
                    top_comment = item["snippet"]["topLevelComment"]["snippet"]
                    comment_info = {
                        "author": top_comment.get("authorDisplayName"),
                        "text": top_comment.get("textDisplay"),
                        "likeCount": top_comment.get("likeCount"),
                        "publishedAt": top_comment.get("publishedAt"),
                        "replies": []
                    }
                    
                    # 답글(대댓글) 정보
                    if "replies" in item:
                        for reply in item["replies"]["comments"]:
                            reply_snippet = reply["snippet"]
                            comment_info["replies"].append({
                                "author": reply_snippet.get("authorDisplayName"),
                                "text": reply_snippet.get("textDisplay"),
                                "likeCount": reply_snippet.get("likeCount"),
                                "publishedAt": reply_snippet.get("publishedAt")
                            })
                    
                    comments_data.append(comment_info)
                
                # 다음 페이지 확인
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                
                time.sleep(0.5) # API 호출 간 딜레이
                    
            except HttpError as e:
                if "disabled comments" in str(e):
                    print(f"  [알림] 동영상(ID: {video_id})의 댓글이 비활성화되어 있습니다.")
                else:
                    print(f"  [오류] 댓글 스레드 조회 중 오류 발생 (ID: {video_id}): {e}")
                break
                
        return comments_data

    def crawl_video(self, video_id):
        """특정 비디오의 정보를 수집하여 DataFrame으로 반환"""
        if not self.youtube: return None

        # 1. 상세 정보
        videos = self.get_video_details([video_id])
        if not videos: return None
        video = videos[0]

        # 2. 댓글 수집
        comments = []
        if video['commentCount'] > 0:
            comments = self.get_comment_threads(video_id)
        
        # 3. Flatten to Rows
        all_rows = []
        video_base_meta = {
            "video_id": video['id'],
            "video_title": video['title'],
            "video_published_at": video['publishedAt'],
            "video_view_count": video['viewCount']
        }

        if not comments:
            all_rows.append({
                "source": "YouTube",
                "author": "",
                "date": "",
                "rating": "",
                "contents": "",
                "metadata": format_metadata(video_base_meta)
            })
        else:
            for comment in comments:
                # Top comment
                meta = video_base_meta.copy()
                meta["like_count"] = comment.get('likeCount')
                meta["is_reply"] = False
                
                all_rows.append({
                    "source": "YouTube",
                    "author": comment.get('author', ''),
                    "date": comment.get('publishedAt', ''),
                    "rating": "",
                    "contents": comment.get('text', ''),
                    "metadata": format_metadata(meta)
                })
                
                # Replies
                for reply in comment.get('replies', []):
                    rep_meta = video_base_meta.copy()
                    rep_meta["like_count"] = reply.get('likeCount')
                    rep_meta["is_reply"] = True
                    
                    all_rows.append({
                        "source": "YouTube",
                        "author": reply.get('author', ''),
                        "date": reply.get('publishedAt', ''),
                        "rating": "",
                        "contents": reply.get('text', ''),
                        "metadata": format_metadata(rep_meta)
                    })

        return pd.DataFrame(all_rows)

class CoupangParser:
    def parse_html(self, html_doc):
        soup = BeautifulSoup(html_doc, 'html.parser')
        all_reviews_data = []

        # 1. <article> 태그 찾기 (class 이름은 변경될 수 있으므로 유연하게)
        # 기존 코드의 선택자: article.twc-pt-[16px]...
        articles = soup.select('article')

        if not articles:
            return pd.DataFrame()

        for article in articles:
            try:
                # 사용자 이름
                user_el = article.select_one('span[data-member-id]')
                author = user_el.text.replace('\xa0', '').strip() if user_el else "Unknown"

                # 별점
                stars_el = article.find_all('i', class_='twc-bg-full-star')
                rating = len(stars_el) if stars_el else ""

                # 작성일
                date_el = article.find(string=re.compile(r'\d{4}\.\d{2}\.\d{2}'))
                date = date_el.strip() if date_el else ""

                # 내용
                content_el = article.select_one('span[translate="no"]')
                contents = content_el.get_text('\n', strip=True) if content_el else ""

                # 구매 옵션
                opt_el = article.select_one('.sdp-review__article__list__info__product-info__name')
                option_name = opt_el.get_text(strip=True) if opt_el else ""

                meta_dict = {}
                if option_name:
                    meta_dict["options"] = option_name

                if contents:
                    all_reviews_data.append({
                        "source": "Coupang",
                        "author": author,
                        "date": date,
                        "rating": str(rating),
                        "contents": contents,
                        "metadata": format_metadata(meta_dict)
                    })

            except Exception as e:
                print(f"Parsing error: {e}")
                continue

        if not all_reviews_data:
            return pd.DataFrame()
            
        return pd.DataFrame(all_reviews_data)

class AmazonParser:
    def parse_html(self, html_doc):
        """아마존 리뷰 페이지 HTML 파싱 (수동 붙여넣기 대응)"""
        soup = BeautifulSoup(html_doc, 'html.parser')
        all_reviews_data = []
        
        # 아마존 리뷰 컨테이너 식별 (div 또는 li)
        reviews = soup.select('div[data-hook="review"]')
        if not reviews:
            reviews = soup.select('li[data-hook="review"]')
            
        if not reviews:
            return pd.DataFrame()
            
        for item in reviews:
            try:
                # 사용자 이름
                author_el = item.select_one('.a-profile-name')
                author = author_el.get_text(strip=True) if author_el else "Unknown"
                
                # 별점 (예: 5.0 out of 5 stars)
                rating_el = item.select_one('i[data-hook="review-star-rating"] span.a-icon-alt')
                if not rating_el:
                    rating_el = item.select_one('i[data-hook="cmps-review-star-rating"] span.a-icon-alt')
                
                rating_text = rating_el.get_text(strip=True) if rating_el else ""
                match = re.search(r'(\d+(\.\d+)?)', rating_text)
                rating = match.group(1) if match else rating_text
                
                # 작성일
                date_el = item.select_one('span[data-hook="review-date"]')
                date = date_el.get_text(strip=True) if date_el else ""
                
                # 옵션 (색상, 스타일 등)
                options_el = item.select_one('a[data-hook="format-strip"]')
                options = options_el.get_text(strip=True) if options_el else ""
                
                # 제목
                title_el = item.select_one('a[data-hook="review-title"]')
                title_text = ""
                if title_el:
                    original_title = title_el.select_one('.cr-original-review-content')
                    if original_title:
                        title_text = original_title.get_text(strip=True)
                    else:
                        title_text = title_el.get_text(" ", strip=True)
                        title_text = re.sub(r'^\d\.\d( out of \d stars|별 \d개 중)?\s*', '', title_text).strip()

                # 본문
                body_el = item.select_one('span[data-hook="review-body"]')
                body_text = ""
                if body_el:
                    original_body = body_el.select_one('.cr-original-review-content')
                    if original_body:
                        body_text = original_body.get_text('\n', strip=True)
                    else:
                        body_text = body_el.get_text('\n', strip=True)
                
                # 통합 내용 구성
                contents = f"{title_text}\n{body_text}".strip()
                
                if contents:
                    meta_dict = {"options": options}
                    all_reviews_data.append({
                        "source": "Amazon",
                        "author": author,
                        "date": date,
                        "rating": str(rating),
                        "contents": contents,
                        "metadata": format_metadata(meta_dict)
                    })

            except Exception as e:
                print(f"Amazon Parsing error: {e}")
                continue
                
        return pd.DataFrame(all_reviews_data)

class SamsungCrawler:
    def __init__(self):
        self.options = Options()
        # self.options.add_argument("--headless")  # 브라우저 창 보고 싶으면 주석 처리 유지 (UI에서 제어 가능하면 좋겠지만 일단 코드 그대로)
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.options)

    def crawl_reviews(self, url, max_pages=13):
        """
        max_pages: 수집할 페이지 수 (15로 설정하면 1~15페이지 수집)
        """
        print(f"🌍 접속 중... {url}")
        self.driver.get(url)
        time.sleep(5)  # 최초 접속 시 충분히 대기

        all_reviews = []
        
        # 반복문: 0부터 max_pages-1 까지 (예: 0~14 -> 총 15회)
        for current_page in range(max_pages):
            try:
                print(f"\n📄 [진행 중] {current_page + 1} 페이지 수집 시작...")

                # 1. 리뷰 리스트 로딩 대기
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.ID, "review-list"))
                    )
                except:
                    print("⚠️ 리뷰 리스트를 찾지 못했습니다. (로딩 지연 또는 리뷰 없음)")
                
                # 2. 데이터 파싱
                html = self.driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                review_items = soup.select("#review-list > li[id^='cmntList']")
                
                if not review_items:
                    print("⚠️ 더 이상 표시할 리뷰가 없습니다.")
                    break

                # 3. 데이터 추출
                extracted_count = 0
                for item in review_items:
                    try:
                        author = item.select_one(".userid").get_text(strip=True) if item.select_one(".userid") else "Unknown"
                        date = item.select_one(".date").get_text(strip=True) if item.select_one(".date") else ""
                        
                        rating_el = item.select_one(".review-starating")
                        rating = "N/A"
                        if rating_el and rating_el.get('aria-label'):
                            rating = re.sub(r'[^0-9]', '', rating_el.get('aria-label'))

                        content_el = item.select_one(".review-text .txt-slide p")
                        content = content_el.get_text(strip=True) if content_el else ""
                        
                        source_el = item.select_one(".buy-source")
                        source_info = source_el.get_text(strip=True) if source_el else ""
                        
                        meta_dict = {
                            "page": current_page + 1,
                            "buy_source": source_info
                        }
                        all_reviews.append({
                            "source": "Samsung.com",
                            "author": author,
                            "date": date,
                            "rating": str(rating),
                            "contents": content,
                            "metadata": format_metadata(meta_dict)
                        })
                        extracted_count += 1
                    except Exception as e:
                        continue
                
                print(f"   ✅ {extracted_count}개 리뷰 저장 완료 (누적 {len(all_reviews)}개)")

                # 4. 다음 페이지 이동 (마지막 페이지가 아니면)
                if current_page < max_pages - 1:
                    try:
                        # '다음' 버튼 찾기 (화살표 아이콘)
                        next_btn = WebDriverWait(self.driver, 5).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, ".paging .btn_next a"))
                        )
                        
                        # 버튼이 유효한지 확인
                        href = next_btn.get_attribute("href")
                        if not href or "javascript" not in href:
                            print("🛑 마지막 페이지에 도달했습니다.")
                            break

                        # 클릭 실행 (JavaScript로 강제 클릭하여 오류 방지)
                        self.driver.execute_script("arguments[0].click();", next_btn)
                        
                        # 중요: 페이지 로딩 대기 (이 시간이 짧으면 이전 페이지 데이터를 또 긁습니다)
                        print("   rabbit... 다음 페이지 로딩 중 (3초 대기)")
                        time.sleep(3) 
                        
                    except Exception as e:
                        print("🛑 다음 페이지 버튼을 찾을 수 없습니다. (마지막 페이지 추정)")
                        break

            except Exception as e:
                print(f"❌ {current_page + 1}페이지 처리 중 오류: {e}")
                break

        return pd.DataFrame(all_reviews)
    
    def close(self):
        try:
            self.driver.quit()
        except:
            pass

class LGCrawler:
    def __init__(self):
        self.options = Options()
        # self.options.add_argument("--headless") # 브라우저 창 숨기기
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.options)

    def crawl_reviews(self, url, max_more_clicks=5):
        """
        max_more_clicks: '리뷰 더보기' 버튼을 클릭할 횟수
        """
        print(f"🌍 LG전자 접속 중... {url}")
        self.driver.get(url)
        time.sleep(5) # 페이지 로딩 대기

        # 1. 리뷰 탭이나 섹션으로 이동 (스크롤)
        print("🖱️ 리뷰 섹션으로 스크롤 이동...")
        try:
            # 리뷰 영역이 보일 때까지 스크롤 (id="reviewArea" 또는 "divReviewList")
            review_area = self.driver.find_element(By.ID, "reviewArea")
            self.driver.execute_script("arguments[0].scrollIntoView(true);", review_area)
            time.sleep(2)
        except:
            # 못 찾으면 그냥 스크롤 조금 내림
            self.driver.execute_script("window.scrollBy(0, 1000);")
            time.sleep(2)

        # 2. '리뷰 더보기' 버튼 반복 클릭하여 리스트 확장
        click_count = 0
        while click_count < max_more_clicks:
            try:
                # 더보기 버튼 찾기 (제공해주신 HTML 기준 id="reviewMoreBtn")
                more_btn = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.ID, "reviewMoreBtn"))
                )
                
                # 버튼 클릭 (JS로 강제 클릭이 더 안정적)
                self.driver.execute_script("arguments[0].click();", more_btn)
                print(f"   [+] '리뷰 더보기' 클릭 {click_count + 1}회 성공")
                
                # 데이터 로딩 대기 (중요)
                time.sleep(2) 
                click_count += 1
                
            except Exception:
                print("   🚫 더 이상 '리뷰 더보기' 버튼이 없거나 클릭할 수 없습니다.")
                break

        # 3. 확장된 전체 HTML 파싱
        print("\n📄 전체 리뷰 데이터 추출 시작...")
        html = self.driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        # 리뷰 리스트 가져오기 (id="divReviewList" > li)
        review_items = soup.select("#divReviewList > li")
        
        all_reviews = []
        
        for item in review_items:
            try:
                # [작성자] class="user-name" (안에 '구매자 이름' 텍스트 제거)
                author_el = item.select_one(".user-name")
                author = author_el.get_text(strip=True).replace("구매자 이름", "") if author_el else "Unknown"

                # [작성일] class="purchase-date" (안에 '구매 일자' 텍스트 제거)
                date_el = item.select_one(".purchase-date")
                date = date_el.get_text(strip=True).replace("구매 일자", "") if date_el else ""

                # [별점] class="score-wrap" -> "blind" 태그 안의 텍스트 (예: "리뷰 별점 5점 중 5점")
                rating = "N/A"
                rating_el = item.select_one(".score-wrap .blind")
                if rating_el:
                    # 정규식으로 숫자 중 가장 마지막 숫자(5점 중 '5'점) 추출 또는 '5점' 찾기
                    nums = re.findall(r'\d+', rating_el.get_text())
                    if nums:
                        rating = nums[-1] # 마지막 숫자가 실제 점수일 확률 높음

                # [내용] class="message-wrap" -> "message"
                content_el = item.select_one(".message-wrap .message")
                content = content_el.get_text(strip=True) if content_el else ""

                # [옵션/구매처] 
                # 구매처: .purchase-path
                source_path = ""
                path_el = item.select_one(".purchase-path")
                if path_el:
                    source_path = path_el.get_text(strip=True).replace("구매경로", "")
                
                # 옵션: .option-list -> dd
                options = []
                option_els = item.select(".option-list dd")
                for opt in option_els:
                    options.append(opt.get_text(strip=True))
                option_str = ", ".join(options)

                # 데이터 저장
                meta_dict = {
                    "buy_source": source_path,
                    "options": option_str
                }
                all_reviews.append({
                    "source": "LGE.co.kr",
                    "author": author,
                    "date": date,
                    "rating": str(rating),
                    "contents": content,
                    "metadata": format_metadata(meta_dict)
                })

            except Exception as e:
                print(f"   ❌ 리뷰 항목 파싱 에러: {e}")
                continue

        return pd.DataFrame(all_reviews)

    def close(self):
        try:
            self.driver.quit()
        except:
            pass

class CoupangCrawler:
    def __init__(self):
        self.options = Options()
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36")
        self.options.add_argument("--disable-blink-features=AutomationControlled")
        self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.options.add_experimental_option('useAutomationExtension', False)
        
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.options)

    def crawl_reviews(self, url, max_pages=5):
        """
        max_pages: 수집할 페이지 수
        """
        print(f"🚀 Coupang 접속 중... {url}")
        self.driver.get(url)
        time.sleep(3) 

        # 1. '상품평' 탭 클릭 (필요한 경우)
        try:
            # 상품평 탭 찾기 (상품 페이지의 경우)
            review_tab = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.ID, "btfTab-review")) # id="btfTab-review" 가 일반적임
            )
            self.driver.execute_script("arguments[0].click();", review_tab)
            time.sleep(1)
        except:
             # 바로 들어온 곳이 리뷰 페이지거나 구조가 다를 수 있음
            pass

        # 스크롤 조금 내려서 리뷰 영역 로드 유도
        self.driver.execute_script("window.scrollBy(0, 1000);")
        time.sleep(2)

        all_reviews = []

        for page in range(1, max_pages + 1):
            print(f"\n📄 [진행 중] {page} 페이지 수집 중...")
            
            # 페이지 로딩 확인 (리뷰 리스트)
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "article.sdp-review__article__list"))
                )
            except:
                print("⚠️ 리뷰 리스트를 찾을 수 없습니다.")
                break

            # 데이터 파싱
            html = self.driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            review_items = soup.select("article.sdp-review__article__list")

            if not review_items:
                print("⚠️ 리뷰 데이터가 없습니다.")
                break

            extracted_cnt = 0
            for item in review_items:
                try:
                    # 작성자
                    author = item.select_one(".sdp-review__article__list__info__user__name")
                    author_text = author.get_text(strip=True) if author else "Unknown"

                    # 별점 (data-rating 속성)
                    rating_el = item.select_one(".sdp-review__article__list__info__product-info__star-orange")
                    rating = rating_el.get("data-rating") if rating_el else "N/A"

                    # 날짜
                    date_el = item.select_one(".sdp-review__article__list__info__product-info__reg-date")
                    date = date_el.get_text(strip=True) if date_el else ""

                    # 옵션명
                    opt_el = item.select_one(".sdp-review__article__list__info__product-info__name")
                    option_name = opt_el.get_text(strip=True) if opt_el else ""

                    # 내용 (헤드라인 + 본문)
                    headline_el = item.select_one(".sdp-review__article__list__headline")
                    content_el = item.select_one(".sdp-review__article__list__review__content")
                    
                    full_content = []
                    if headline_el: full_content.append(headline_el.get_text(strip=True))
                    if content_el: full_content.append(content_el.get_text(strip=True))
                    
                    final_content = " ".join(full_content)

                    meta_dict = {
                        "page": page,
                        "options": option_name
                    }
                    all_reviews.append({
                        "source": "Coupang",
                        "author": author_text,
                        "date": date,
                        "rating": str(rating),
                        "contents": final_content,
                        "metadata": format_metadata(meta_dict)
                    })
                    extracted_cnt += 1

                except Exception as e:
                    continue
            
            print(f"   ✅ {extracted_cnt}개 추출 (누적 {len(all_reviews)}개)")

            # 다음 페이지 클릭
            # 쿠팡 페이지네이션: <button class="sdp-review__article__page__num js_reviewArticlePageBtn" data-page="2">2</button>
            # 10페이지 넘어가서 [다음 >] 버튼 처리 필요하지만 일단 1~10페이지 내 이동 구현
            if page < max_pages:
                try:
                    next_page_num = page + 1
                    # 다음 페이지 버튼 찾기 (텍스트 숫자가 일치하는 버튼)
                    # data-page 속성을 이용하여 정확히 타겟팅
                    next_btn = self.driver.find_element(By.CSS_SELECTOR, f"button.sdp-review__article__page__num[data-page='{next_page_num}']")
                    
                    # 버튼 클릭
                    self.driver.execute_script("arguments[0].click();", next_btn)
                    time.sleep(3) # 로딩 대기
                    
                    # 10페이지 단위 넘어가는 '다음' 화살표 처리 등은 복잡해지므로 여기서는 간단히 숫자 버튼만 처리
                    # 만약 숫자 버튼이 안보이면(11페이지 등) -> '다음' 화살표 로직 필요할 수 있음
                except:
                    # 숫자 버튼 못찾으면 다음 화살표 시도 (next-page-btn)
                    try:
                        next_arrow = self.driver.find_element(By.CSS_SELECTOR, ".sdp-review__article__page__next")
                        self.driver.execute_script("arguments[0].click();", next_arrow)
                        time.sleep(3)
                    except:
                        print("🛑 다음 페이지 버튼을 찾을 수 없습니다.")
                        break

        return pd.DataFrame(all_reviews)

    def close(self):
        try:
            self.driver.quit()
        except:
            pass

class BestBuyParser:
    def parse_html(self, html_doc):
        """Best Buy 리뷰 HTML 파싱 (수동 붙여넣기 대응)"""
        soup = BeautifulSoup(html_doc, 'html.parser')
        all_reviews_data = []

        # Best Buy 리뷰 컨테이너 식별
        # 1. 일반적인 리스트 아이템 (li.review-item)
        reviews = soup.select('li.review-item')
        if not reviews:
            # 2. 다른 디자인 대응 (div.ugc-review)
            reviews = soup.select('.ugc-review')
        if not reviews:
            # 3. 더 일반적인 클래스
            reviews = soup.select('.review-list > li')

        if not reviews:
            return pd.DataFrame()

        for item in reviews:
            review_data = {"source": "Best Buy"}
            try:
                # 사용자 이름
                author_el = item.select_one('.ugc-author')
                if not author_el: author_el = item.select_one('.author-name')
                review_data['author'] = author_el.get_text(strip=True) if author_el else "Unknown"

                # 별점
                rating_el = item.select_one('.c-review-average')
                rating_text = rating_el.get_text(strip=True) if rating_el else ""
                match = re.search(r'(\d)', rating_text)
                rating = match.group(1) if match else "N/A"
                if rating == "N/A":
                    # svg title or aria-label check
                    star_el = item.select_one('.c-review-average svg') # number of stars?
                    pass 
                review_data['rating'] = rating

                # 작성일
                date_el = item.select_one('.submission-date')
                review_data['date'] = date_el.get_text(strip=True) if date_el else ""

                # 내용 (제목 + 본문)
                title_el = item.select_one('.ugc-review-title')
                if not title_el: title_el = item.select_one('.review-title')
                title_text = title_el.get_text(strip=True) if title_el else ""

                body_el = item.select_one('.ugc-review-body')
                if not body_el: body_el = item.select_one('.review-text')
                body_text = body_el.get_text('\n', strip=True) if body_el else ""

                contents = f"{title_text}\n{body_text}".strip()

                # 옵션 (Verified Purchase 등)
                verified_el = item.select_one('.verified-purchase')
                options = "Verified Purchase" if verified_el else ""

                if contents:
                    meta_dict = {"options": options}
                    all_reviews_data.append({
                        "source": "Best Buy",
                        "author": review_data.get('author', 'Unknown'),
                        "date": review_data.get('date', ''),
                        "rating": str(rating),
                        "contents": contents,
                        "metadata": format_metadata(meta_dict)
                    })

            except Exception as e:
                print(f"Best Buy Parsing error: {e}")
                continue

        return pd.DataFrame(all_reviews_data)