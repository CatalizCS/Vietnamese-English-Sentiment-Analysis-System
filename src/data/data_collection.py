from typing import List, Dict
import pandas as pd
import os
import json
import datetime
from tqdm import tqdm
from google_play_scraper import Sort, reviews
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import requests

class DataCollector:
    def __init__(self, config):
        self.config = config
        self.output_dir = os.path.join(config.DATA_DIR, "raw_data")
        os.makedirs(self.output_dir, exist_ok=True)

    def collect_google_play_reviews(self, app_id: str, language: str = 'vi', sample_counts: Dict[str, int] = None) -> pd.DataFrame:
        """Collect reviews from Google Play with custom sample counts"""
        try:
            # Calculate total samples needed
            total_samples = 1000  # Default
            if sample_counts:
                total_samples = sum(sample_counts.values())

            # Collect extra reviews to ensure we have enough after filtering
            buffer_multiplier = 2
            result, _ = reviews(
                app_id,
                lang=language,
                country='vn',
                sort=Sort.NEWEST,
                count=total_samples * buffer_multiplier
            )
            
            df = pd.DataFrame(result)
            if df.empty:
                return pd.DataFrame()

            # Map scores to sentiment categories
            df['sentiment'] = df['score'].apply(
                lambda x: 2 if x > 3 else 0 if x < 3 else 1
            )
            
            # Balance dataset according to sample_counts if provided
            if sample_counts:
                balanced_dfs = []
                for sentiment, count in sample_counts.items():
                    sentiment_val = 2 if sentiment == 'positive' else 0 if sentiment == 'negative' else 1
                    sentiment_df = df[df['sentiment'] == sentiment_val]
                    if len(sentiment_df) > count:
                        sentiment_df = sentiment_df.sample(n=count)
                    balanced_dfs.append(sentiment_df)
                df = pd.concat(balanced_dfs)

            # Prepare final dataframe
            df = df[['content', 'sentiment']].rename(columns={'content': 'text'})
            return df

        except Exception as e:
            print(f"Error collecting Google Play reviews: {str(e)}")
            return pd.DataFrame()

    def collect_shopee_reviews(self, product_ids: List[str], sample_counts: Dict[str, int] = None) -> pd.DataFrame:
        """Collect reviews from Shopee with sample count control"""
        reviews_data = []
        
        for product_id in product_ids:
            try:
                url = f"https://shopee.vn/api/v2/item/get_ratings?itemid={product_id}&limit={max_reviews}"
                response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                data = response.json()
                
                for review in data.get('data', {}).get('ratings', []):
                    rating = review.get('rating_star', 0)
                    comment = review.get('comment', '')
                    
                    if comment:
                        reviews_data.append({
                            'text': comment,
                            'label': 2 if rating > 3 else 0 if rating < 3 else 1,
                            'source': 'shopee'
                        })
            except Exception as e:
                print(f"Lỗi khi thu thập dữ liệu Shopee: {str(e)}")
                continue
                
        return pd.DataFrame(reviews_data)

    def collect_facebook_comments(self, post_ids: List[str], access_token: str, sample_counts: Dict[str, int] = None) -> pd.DataFrame:
        """Collect comments from Facebook with sample count control"""
        comments_data = []
        
        for post_id in post_ids:
            try:
                url = f"https://graph.facebook.com/v12.0/{post_id}/comments"
                params = {
                    'access_token': access_token,
                    'limit': 100
                }
                
                response = requests.get(url, params=params)
                data = response.json()
                
                for comment in data.get('data', []):
                    comments_data.append({
                        'text': comment.get('message', ''),
                        'source': 'facebook',
                        'label': None  # Cần gán nhãn thủ công
                    })
            except Exception as e:
                print(f"Lỗi khi thu thập dữ liệu Facebook: {str(e)}")
                continue
                
        return pd.DataFrame(comments_data)

    def collect_manual_reviews(self, input_file: str, sample_counts: Dict[str, int] = None) -> pd.DataFrame:
        """Import reviews from file with sample count control"""
        try:
            if input_file.endswith('.csv'):
                df = pd.read_csv(input_file)
            else:
                df = pd.read_excel(input_file)

            if 'sentiment' not in df.columns and 'label' in df.columns:
                df['sentiment'] = df['label']

            if sample_counts:
                balanced_dfs = []
                for sentiment, count in sample_counts.items():
                    sentiment_val = 2 if sentiment == 'positive' else 0 if sentiment == 'negative' else 1
                    sentiment_df = df[df['sentiment'] == sentiment_val]
                    if len(sentiment_df) > count:
                        sentiment_df = sentiment_df.sample(n=count)
                    balanced_dfs.append(sentiment_df)
                df = pd.concat(balanced_dfs)

            return df[['text', 'sentiment']]

        except Exception as e:
            print(f"Error loading manual reviews: {str(e)}")
            return pd.DataFrame()

    def save_collected_data(self, df: pd.DataFrame, source: str):
        """Lưu dữ liệu đã thu thập"""
        if not df.empty:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                self.output_dir,
                f"{source}_reviews_{timestamp}.csv"
            )
            df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"Đã lưu {len(df)} đánh giá vào {output_file}")