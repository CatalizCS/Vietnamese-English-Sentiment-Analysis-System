
from src.config import Config
from src.data.data_collection import DataCollector
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Thu thập dữ liệu đánh giá')
    parser.add_argument('--source', type=str, required=True,
                      choices=['google_play', 'shopee', 'facebook', 'manual'],
                      help='Nguồn dữ liệu cần thu thập')
    parser.add_argument('--app_id', type=str, help='ID ứng dụng Google Play')
    parser.add_argument('--product_ids', type=str, help='ID sản phẩm Shopee (phân cách bằng dấu phẩy)')
    parser.add_argument('--post_ids', type=str, help='ID bài đăng Facebook (phân cách bằng dấu phẩy)')
    parser.add_argument('--access_token', type=str, help='Facebook API access token')
    parser.add_argument('--input_file', type=str, help='Đường dẫn file Excel/CSV chứa đánh giá')
    parser.add_argument('--count', type=int, default=100, help='Số lượng đánh giá cần thu thập')
    
    args = parser.parse_args()
    config = Config()
    collector = DataCollector(config)
    
    if args.source == 'google_play':
        if not args.app_id:
            print("Vui lòng cung cấp app_id")
            return
        df = collector.collect_google_play_reviews(args.app_id, count=args.count)
        
    elif args.source == 'shopee':
        if not args.product_ids:
            print("Vui lòng cung cấp product_ids")
            return
        product_ids = args.product_ids.split(',')
        df = collector.collect_shopee_reviews(product_ids, max_reviews=args.count)
        
    elif args.source == 'facebook':
        if not args.post_ids or not args.access_token:
            print("Vui lòng cung cấp post_ids và access_token")
            return
        post_ids = args.post_ids.split(',')
        df = collector.collect_facebook_comments(post_ids, args.access_token)
        
    elif args.source == 'manual':
        if not args.input_file:
            print("Vui lòng cung cấp input_file")
            return
        df = collector.collect_manual_reviews(args.input_file)
    
    if not df.empty:
        collector.save_collected_data(df, args.source)

if __name__ == "__main__":
    main()