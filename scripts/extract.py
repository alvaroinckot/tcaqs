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

# Use lxml parser for better performance
def extract_info(info):
    # Try to parse with lxml for speed, fallback to html.parser if needed
    try:
        soup = beauty(info, features="lxml")
    except Exception as e:
        # Fallback to default parser if lxml fails
        soup = beauty(info, features="html.parser")
    
    try:
        header_divs = soup.find_all('div',['AuctionHeader'])
        if not header_divs:
            raise ValueError("No auction header found in HTML")
        
        header_text = header_divs[0].get_text()
        split_header = header_text.split("|")
        
        if len(split_header) < 4:
            raise ValueError(f"Invalid header format: {header_text[:50]}...")
            
        [char_raw, vocation_raw, gender_raw, server_raw] = split_header
    except Exception as e:
        raise ValueError(f"Failed to parse auction header: {str(e)}")

    short_auction_data_value =  soup.find_all('div',['ShortAuctionDataValue'])

    status = name = soup.find_all('div',['AuctionInfo'])[0].get_text()
    name = soup.find_all('div',['AuctionCharacterName'])[0].get_text()
    bid = int(short_auction_data_value[2].get_text().replace(',', '').strip())
    level = int(char_raw.split(':')[1].strip())
    vocation = vocation_raw.split(':')[1].strip().split(' ')[-1].lower()
    server = server_raw.split(':')[1].strip()
    is_name_contains_special_character = "'" in name
    levels = soup.select('.LevelColumn')
    axe_fighting = int(levels[0].text)
    club_fighting = int(levels[1].text)
    distance_fighting = int(levels[2].text)
    fishing = int(levels[3].text)
    fist_fighting = int(levels[4].text)
    magic_level = int(levels[5].text)
    shielding = int(levels[6].text)
    sword_fighting = int(levels[7].text)

    general_details = soup.select("#CharacterDetailsGeneral div")

    mounts = int(general_details[5].text)
    outfits = int(general_details[6].text)

    character_details = soup.select("#CharacterDetails div div")

    gold = int(character_details[34].text.replace(",",""))
    achievement_points = int(character_details[35].text.replace(",",""))
    is_transfer_available = 'used immediately' in str(character_details[36].text)
    available_charm_points = int(character_details[38].text.replace(",",""))
    spent_charm_points = int(character_details[39].text.replace(",",""))
    charm_expansion = 'yes' in str(character_details[37].text)
    hunting_task_points = int(character_details[41].text.replace(",",""))
    permanent_prey_task_slot = int(character_details[42].text)
    permanent_hunt_task_slot = int(character_details[43].text)
    prey_wildcards = int(character_details[44].text)
    hirelings = int(character_details[45].text)
    hirelings_jobs = int(character_details[46].text)
    hirelings_outfits = int(character_details[47].text)


    auction_start_date = datetime.datetime.strptime(short_auction_data_value[0].get_text().replace("\xa0", " ").replace("CEST", "").replace("CST", "").replace("CET", "").strip(), "%b %d %Y, %H:%M")
    auction_end_date = datetime.datetime.strptime(short_auction_data_value[1].get_text().replace("\xa0", " ").replace("CEST", "").replace("CST", "").replace("CET", "").strip(), "%b %d %Y, %H:%M")


    imbuements_table = soup.select('#Imbuements td td td')
    imbuments = len(imbuements_table) - 1

    charms_table = soup.select('#Charms td td td')
    charms = int(len(charms_table) / 2 - 1)

    return {
        'status': status,
        'name': name,
        'bid': bid,
        'level': level,
        'vocation': vocation,
        'server': server,
        'is_name_contains_special_character': is_name_contains_special_character,
        'axe_fighting': axe_fighting,
        'club_fighting': club_fighting,
        'distance_fighting': distance_fighting,
        'fishing': fishing,
        'fist_fighting': fist_fighting,
        'magic_level': magic_level,
        'shielding': shielding,
        'sword_fighting': sword_fighting,
        'mounts': mounts,
        'outfits': outfits,
        'gold': gold,
        'achievement_points': achievement_points,
        'is_transfer_available': is_transfer_available,
        'available_charm_points': available_charm_points,
        'spent_charm_points': spent_charm_points,
        'charm_expansion': charm_expansion,
        'hunting_task_points': hunting_task_points,
        'permanent_prey_task_slot': permanent_prey_task_slot,
        'permanent_hunt_task_slot': permanent_hunt_task_slot,
        'prey_wildcards': prey_wildcards,
        'hirelings': hirelings,
        'hirelings_jobs': hirelings_jobs,
        'hirelings_outfits': hirelings_outfits,
        'auction_start_date_iso': auction_start_date.isoformat(),
        'auction_end_date_iso': auction_end_date.isoformat(),
        'imbuements': imbuments,
        'charms': charms
    }


# Connect to SQLite database (will create if doesn't exist)
conn = sqlite3.connect('characters.db')
cursor = conn.cursor()

# Create table if it doesn't exist with optimized structure
cursor.execute('''
CREATE TABLE IF NOT EXISTS characters (
    id INTEGER PRIMARY KEY,
    name TEXT,
    status TEXT,
    bid INTEGER,
    level INTEGER,
    vocation TEXT,
    server TEXT,
    is_name_contains_special_character BOOLEAN,
    axe_fighting INTEGER,
    club_fighting INTEGER,
    distance_fighting INTEGER,
    fishing INTEGER,
    fist_fighting INTEGER,
    magic_level INTEGER,
    shielding INTEGER,
    sword_fighting INTEGER,
    mounts INTEGER,
    outfits INTEGER,
    gold INTEGER,
    achievement_points INTEGER,
    is_transfer_available BOOLEAN,
    available_charm_points INTEGER,
    spent_charm_points INTEGER,
    charm_expansion BOOLEAN,
    hunting_task_points INTEGER,
    permanent_prey_task_slot INTEGER,
    permanent_hunt_task_slot INTEGER,
    prey_wildcards INTEGER,
    hirelings INTEGER,
    hirelings_jobs INTEGER,
    hirelings_outfits INTEGER,
    auction_start_date_iso TEXT,
    auction_end_date_iso TEXT,
    imbuements INTEGER,
    charms INTEGER
)
''')

# Create indexes for common query fields
cursor.execute('CREATE INDEX IF NOT EXISTS idx_vocation ON characters(vocation)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_level ON characters(level)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_server ON characters(server)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_bid ON characters(bid)')

html_dir = Path("$BASE_PATH/Downloads/scrap/scrap14/scrap")

html_files = list(html_dir.glob("*.html"))
print(f"Found {len(html_files)} HTML files to process")

# Function to extract data from files (without DB operations)
def extract_from_file(file_index_and_path):
    index, file_path = file_index_and_path
    auction_id = int(file_path.stem)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        data = extract_info(html_content)
        data['id'] = auction_id
        return data
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return None

# Process a batch of files and return their data
def process_file_batch(file_batch):
    results = []
    for file_data in file_batch:
        data = extract_from_file(file_data)
        if data:
            results.append(data)
    return results

# Function to save a batch of data to the database
def save_batch_to_db(data_batch, db_path='characters.db'):
    if not data_batch:
        return 0

    # Use one connection per batch
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Prepare data for bulk insert
    values = []
    for data in data_batch:
        if data:
            values.append((
                data['id'], data['name'], data['status'], data['bid'], data['level'], data['vocation'], data['server'],
                data['is_name_contains_special_character'], data['axe_fighting'], data['club_fighting'], 
                data['distance_fighting'], data['fishing'], data['fist_fighting'], data['magic_level'], 
                data['shielding'], data['sword_fighting'], data['mounts'], data['outfits'], data['gold'],
                data['achievement_points'], data['is_transfer_available'], data['available_charm_points'],
                data['spent_charm_points'], data['charm_expansion'], data['hunting_task_points'],
                data['permanent_prey_task_slot'], data['permanent_hunt_task_slot'], data['prey_wildcards'],
                data['hirelings'], data['hirelings_jobs'], data['hirelings_outfits'], 
                data['auction_start_date_iso'], data['auction_end_date_iso'], data['imbuements'], data['charms']
            ))
    
    # Begin transaction for bulk insert
    conn.execute('BEGIN TRANSACTION')
    cursor.executemany('''
    INSERT OR REPLACE INTO characters VALUES (
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
    )
    ''', values)
    conn.commit()
    conn.close()
    
    return len(values)


# Configuration parameters
BATCH_SIZE = 1000  # Process this many files before saving to DB
MAX_WORKERS = min(os.cpu_count(), multiprocessing.cpu_count())  # Use all available cores
DB_WRITER_PROCESSES = 2  # Number of processes dedicated to writing to the DB
PROCESSING_CHUNK_SIZE = min(10000, len(html_files))  # Size of chunks for distribution

print(f"Using {MAX_WORKERS} worker processes")
print(f"File batch size: {BATCH_SIZE}")
print(f"Processing chunk size: {PROCESSING_CHUNK_SIZE}")

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Create an optimized SQLite database (this helps with performance)
conn.execute('PRAGMA journal_mode = WAL')  # Write-Ahead Logging
conn.execute('PRAGMA synchronous = NORMAL')
conn.execute('PRAGMA cache_size = 1000000')  # Use more memory for cache
conn.execute('PRAGMA temp_store = MEMORY')  # Store temp tables in memory
conn.commit()
conn.close()

# Function to process a batch of files using parallel processing and save the results
def process_chunks_with_multiprocessing(file_list):
    # Create batch chunks
    batches = list(chunks(list(enumerate(file_list)), BATCH_SIZE))
    
    start_time = time.time()
    total_processed = 0
    total_saved = 0
    
    # Create a progress bar for batch processing
    with tqdm.tqdm(total=len(batches), desc="Processing batches", position=0, leave=True) as pbar:
        for i, batch in enumerate(batches):
            # Extract data using process pool
            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                batch_results = []
                # Split the batch for parallel processing
                sub_batches = list(chunks(batch, max(1, len(batch) // MAX_WORKERS)))
                future_to_batch = {executor.submit(process_file_batch, sub_batch): sub_batch for sub_batch in sub_batches}
                
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_results.extend(future.result())
            
            # Save batch results to database
            saved_count = save_batch_to_db(batch_results)
            total_processed += len(batch)
            total_saved += saved_count
            
            # Update progress
            pbar.update(1)
            elapsed = time.time() - start_time
            files_per_sec = total_processed / elapsed if elapsed > 0 else 0
            est_total_time = len(file_list) / files_per_sec if files_per_sec > 0 else 0
            pbar.set_postfix({
                'Files/sec': f'{files_per_sec:.1f}',
                'Saved': total_saved,
                'Est. Total': f'{est_total_time/60:.1f}min'
            })
    
    return total_saved

# Process files in chunks for better memory management
total_saved = 0
for chunk_idx, file_chunk in enumerate(chunks(html_files, PROCESSING_CHUNK_SIZE)):
    print(f"Processing chunk {chunk_idx+1}/{(len(html_files) + PROCESSING_CHUNK_SIZE - 1) // PROCESSING_CHUNK_SIZE} "
          f"({len(file_chunk)} files)")
    chunk_saved = process_chunks_with_multiprocessing(file_chunk)
    total_saved += chunk_saved

# Final stats
conn = sqlite3.connect('characters.db')
cursor = conn.cursor()
final_count = cursor.execute('SELECT COUNT(*) FROM characters').fetchone()[0]
conn.close()

print(f"Total characters processed: {total_saved}")
print(f"Final database count: {final_count}")
