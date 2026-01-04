import json
import os

def convert_json(source_path, template_path, output_path):
    # Load source and template
    with open(source_path, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    with open(template_path, 'r', encoding='utf-8') as f:
        template_data = json.load(f)

    # Prepare the result dictionary based on template structure
    result = {}
    
    source_fields = source_data.get('fields', {})
    
    for key in template_data.keys():
        if key == 'line_items':
            result['line_items'] = []
            source_items = source_data.get('line_items', [])
            template_item_keys = template_data['line_items'][0].keys() if template_data['line_items'] else []
            
            for item in source_items:
                new_item = {}
                for item_key in template_item_keys:
                    # Mapping known field names
                    if item_key == 'description':
                        new_item['description'] = item.get('desc', '')
                    else:
                        new_item[item_key] = item.get(item_key, '')
                result['line_items'].append(new_item)
        else:
            # Try to get from fields or top level
            val = source_fields.get(key)
            if val is None:
                val = source_data.get(key, '')
            result[key] = val

    # Write the output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Conversion complete. Saved to {output_path}")

if __name__ == "__main__":
    base_dir = "/Users/pat/Desktop/custom_FM/working/comparison"
    source_dir = os.path.join(base_dir, "ground_truth")
    template = os.path.join(base_dir, "result/output.json")
    output_dir = os.path.join(source_dir, "converted")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith(".json") and filename != "MSV-001_converted.json": # Skip existing converted file if any
            source_path = os.path.join(source_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            print(f"Processing {filename}...")
            try:
                convert_json(source_path, template, output_path)
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

