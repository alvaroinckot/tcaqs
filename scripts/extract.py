from bs4 import BeautifulSoup as beauty
import cloudscraper
from urllib.parse import urlparse
from urllib.parse import parse_qs
import pandas as pd
from multiprocessing import Pool
import tqdm
import os
import psycopg2
import datetime

servers = pd.read_csv('/home/alvaro/Workspace/sandbox/servers.csv').to_records()

servers_records = {}

for server in servers:
    servers_records[server.name] = server


def extract_info(info):
    soup = beauty(info)
    
    [char_raw, vocation_raw, gender_raw, server_raw] = soup.find_all('div',['AuctionHeader'])[0].get_text().split("|")
    
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
    battleye = servers_records[server].battleye
    pvp_type = servers_records[server].type
    server_location = servers_records[server].location

    general_details = soup.select("#CharacterDetailsGeneral div")
    
    mounts = int(general_details[5].text)
    outfits = int(general_details[6].text)
    
    character_details = soup.select("#CharacterDetails div div")
    
    gold = int(character_details[34].text.replace(",",""))
    achievement_points = int(character_details[35].text)
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
        'battleye': battleye,
        'pvp_type': pvp_type,
        'server_location': server_location,
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


pool = Pool(3)

arr = os.listdir("/home/alvaro/Server/NAS/Alvaro/scrap/")

arr = arr[329365:]

def do_the_magic(file_path):
    try:
        conn = psycopg2.connect("dbname=tcaqs user=tcaqs password=tcaqs host=localhost")
        cur = conn.cursor()
        f = open("/home/alvaro/Server/NAS/Alvaro/scrap/"  + file_path, "r")
        info = f.read()
        data = extract_info(info)
        data['id'] = file_path.replace(".html", "")
        cur.execute("""
        INSERT INTO public.auctions_v3
            (id, status, "name", "bid", "level", vocation, "server", is_name_contains_special_character, axe_fighting, club_fighting, distance_fighting, fishing, fist_fighting, magic_level, shielding, sword_fighting, battleye, pvp_type, server_location, gold, achievement_points, is_transfer_available, available_charm_points, spent_charm_points, charm_expansion, hunting_task_points, permanent_prey_task_slot, permanent_hunt_task_slot, prey_wildcards, hirelings, hirelings_jobs, hirelings_outfits,auction_start_date_iso,auction_end_date_iso,imbuements,charms)
            VALUES({id},'{status}',$${name}$$,{bid},{level},'{vocation}','{server}',{is_name_contains_special_character},{axe_fighting},{club_fighting},{distance_fighting},{fishing},{fist_fighting},{magic_level},{shielding},{sword_fighting},{battleye},'{pvp_type}','{server_location}',{gold},{achievement_points},{is_transfer_available},{available_charm_points},{spent_charm_points},{charm_expansion},{hunting_task_points},{permanent_prey_task_slot},{permanent_hunt_task_slot},{prey_wildcards},{hirelings},{hirelings_jobs},{hirelings_outfits},'{auction_start_date_iso}','{auction_end_date_iso}',{imbuements},{charms})
                ON CONFLICT DO NOTHING;
        """.format(**data))
        conn.commit()
        cur.close()
        conn.close()
    except:
        print("Error with " + file_path)
        f = open("/home/alvaro/processing-errrors-2.txt", "a")
        f.write(file_path + '\n')
        f.close()


print("starting")

results = [x for x in tqdm.tqdm(pool.imap_unordered(
        do_the_magic, arr), total=len(arr)-1)]
                                

pool.close()
pool.join()
   