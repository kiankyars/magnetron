import json
import os
import glob

def json_to_tsv():
    # Get all JSON files in data folder
    json_files = glob.glob("data/out*.json")
    
    for json_file in json_files:
        # Read JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract JSON array (skip the French text at the beginning)
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        
        if json_start == -1 or json_end == 0:
            continue
            
        json_content = content[json_start:json_end]
        data = json.loads(json_content)
        
        # Create TSV filename
        tsv_file = json_file.replace('.json', '.tsv')
        
        # Write TSV file
        with open(tsv_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write("prompt\tsolution0\tsolution1\tlabel\n")
            
            # Write data rows
            for item in data:
                # Escape tabs and newlines in fields
                prompt = item['prompt'].replace('\t', ' ').replace('\n', ' ')
                solution0 = item['solution0'].replace('\t', ' ').replace('\n', ' ')
                solution1 = item['solution1'].replace('\t', ' ').replace('\n', ' ')
                label = str(item['label'])
                
                f.write(f"{prompt}\t{solution0}\t{solution1}\t{label}\n")
        
        print(f"Converted {json_file} to {tsv_file}")

if __name__ == "__main__":
    json_to_tsv()
