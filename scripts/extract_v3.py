from bs4 import BeautifulSoup as beauty
import datetime
import sqlite3
from pathlib import Path
import tqdm
import concurrent.futures
import os
import multiprocessing
import time
import lxml
import json

def check_minimum_bid_auction(auction_id, html_dir="/home/alvaro/Downloads/scrap/scrap14/scrap"):
    """
    Check if an auction has 'Minimum Bid' text in the specified CSS selector.
    Returns True if it's a minimum bid auction (should be marked as failed).
    """
    html_file = Path(html_dir) / f"{auction_id}.html"
    
    if not html_file.exists():
        print(f"Warning: HTML file not found for auction {auction_id}")
        return False
    
    try:
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        soup = beauty(html_content, features="lxml")
        
        # Look for the specific CSS selector
        bid_row_element = soup.select_one(".ShortAuctionDataBidRow > div:nth-child(1)")
        
        if bid_row_element:
            text_content = bid_row_element.get_text().strip()
            return "Minimum Bid" in text_content
        
        return False
        
    except Exception as e:
        print(f"Error processing auction {auction_id}: {e}")
        return False

def process_auction_batch(auction_ids_batch, html_dir="/home/alvaro/Downloads/scrap/scrap14/scrap"):
    """
    Process a batch of auction IDs and return a list of IDs that should be marked as failed.
    """
    failed_auction_ids = []
    
    for auction_id in auction_ids_batch:
        if check_minimum_bid_auction(auction_id, html_dir):
            failed_auction_ids.append(auction_id)
    
    return failed_auction_ids

def update_auctions_to_failed(failed_auction_ids, db_path='characters_v3.db'):
    """
    Update auction status to 'failed' for the given auction IDs.
    """
    if not failed_auction_ids:
        return 0
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Update status to 'failed' for the given IDs
    placeholders = ','.join('?' for _ in failed_auction_ids)
    query = f"UPDATE characters SET status = 'failed' WHERE id IN ({placeholders})"
    
    cursor.execute(query, failed_auction_ids)
    updated_count = cursor.rowcount
    conn.commit()
    conn.close()
    
    return updated_count

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def process_finished_auctions_parallel():
    """
    Main function to process all finished auctions and update failed ones.
    """
    # Configuration
    BATCH_SIZE = 1000  # Process this many auctions before updating DB
    MAX_WORKERS = min(os.cpu_count(), multiprocessing.cpu_count())
    HTML_DIR = "/home/alvaro/Downloads/scrap/scrap14/scrap"
    DB_PATH = 'characters_v3.db'
    
    print(f"Using {MAX_WORKERS} worker processes")
    print(f"Batch size: {BATCH_SIZE}")
    
    # Get all finished auction IDs (including other statuses that should be considered finished)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM characters WHERE status IN ('finished', 'currentlyprocessed', 'will be transferred at the next server save')")
    finished_auction_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    print(f"Found {len(finished_auction_ids)} finished auctions to process")
    
    if not finished_auction_ids:
        print("No finished auctions found.")
        return
    
    # Create batches for processing
    batches = list(chunks(finished_auction_ids, BATCH_SIZE))
    
    start_time = time.time()
    total_processed = 0
    total_failed_updated = 0
    
    # Process batches with progress bar
    with tqdm.tqdm(total=len(batches), desc="Processing batches", position=0, leave=True) as pbar:
        for i, batch in enumerate(batches):
            # Process batch using parallel processing
            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Split the batch for parallel processing
                sub_batches = list(chunks(batch, max(1, len(batch) // MAX_WORKERS)))
                future_to_batch = {executor.submit(process_auction_batch, sub_batch, HTML_DIR): sub_batch 
                                 for sub_batch in sub_batches}
                
                failed_ids_batch = []
                for future in concurrent.futures.as_completed(future_to_batch):
                    failed_ids_batch.extend(future.result())
            
            # Update database with failed auction IDs
            updated_count = update_auctions_to_failed(failed_ids_batch, DB_PATH)
            total_processed += len(batch)
            total_failed_updated += updated_count
            
            # Update progress
            pbar.update(1)
            elapsed = time.time() - start_time
            auctions_per_sec = total_processed / elapsed if elapsed > 0 else 0
            est_total_time = len(finished_auction_ids) / auctions_per_sec if auctions_per_sec > 0 else 0
            pbar.set_postfix({
                'Auctions/sec': f'{auctions_per_sec:.1f}',
                'Failed found': total_failed_updated,
                'Est. Total': f'{est_total_time/60:.1f}min'
            })
    
    # Final statistics
    print(f"\nProcessing completed!")
    print(f"Total auctions processed: {total_processed}")
    print(f"Total auctions updated to 'failed': {total_failed_updated}")
    
    # Verify final counts
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    finished_count = cursor.execute("SELECT COUNT(*) FROM characters WHERE status IN ('finished', 'currentlyprocessed', 'will be transferred at the next server save')").fetchone()[0]
    failed_count = cursor.execute("SELECT COUNT(*) FROM characters WHERE status = 'failed'").fetchone()[0]
    conn.close()
    
    print(f"Remaining 'finished' auctions (including other valid statuses): {finished_count}")
    print(f"Total 'failed' auctions: {failed_count}")

if __name__ == "__main__":
    process_finished_auctions_parallel()
