#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concurrent scoring of generated TTS audio using local prompt templates.
- Load template from prompt file (with placeholder <此处插入待评测的语音风格描述>)
- For each record and instruction type (APS/DSD/RP), insert instruction text into template and call Gemini for scoring
- Parse JSON paragraph from returned text, extract "一致性" field (true/false)
- Stream write: write and flush after each item to prevent data loss

Usage example:
python gemini_eval.py \
  --input_jsonl   generated_results.jsonl \
  --output_jsonl  generated_results_score.jsonl \
  --prompt_file   eval_prompt.txt \
  --api_key       $GENAI_API_KEY \
  --model_name    models/gemini-2.5-pro \
  --instruction_type APS  # Optional: APS, DSD, RP, or ALL
"""
import os
import json
import time
import argparse
import re
import logging
from tqdm import tqdm
from requests.exceptions import HTTPError
from google import genai
from google.genai import types
import multiprocessing as mp
from functools import partial

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='gemini_eval.log',
    filemode='a'
)

def parse_args():
    p = argparse.ArgumentParser(description="Use Gemini to score TTS generated audio for style consistency")
    p.add_argument("--input_jsonl", required=True, help="Input JSONL file path")
    p.add_argument("--output_jsonl", required=True, help="Output JSONL file path")
    p.add_argument("--prompt_file", required=True, help="Prompt template file path with placeholder")
    p.add_argument("--api_key", required=True, help="GenAI API Key")
    p.add_argument("--model_name", default="models/gemini-2.5-pro", help="GenAI model name")
    p.add_argument("--instruction_type", default="ALL", choices=["APS", "DSD", "RP", "ALL"], 
                   help="Only process specified instruction type, ALL means process all types")
    p.add_argument("--num_workers", type=int, default=4, help="Number of concurrent worker processes")
    return p.parse_args()

def load_template(path: str) -> str:
    """Load prompt template"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_json_from_text(text):
    """Extract JSON object from text, handling multiple possible formats"""
    logging.info(f"Original response content:\n{text}")
    
    # Method 1: If JSON is wrapped in markdown code blocks
    try:
        if "```json" in text and "```" in text:
            json_text = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_text:
                parsed = json.loads(json_text.group(1))
                logging.info(f"Successfully extracted using method 1: {parsed}")
                return parsed
    except Exception as e:
        logging.warning(f"Method 1 extraction failed: {e}")
    
    # Method 2: Find JSON objects in text
    try:
        json_pattern = re.compile(r'\{(?:[^{}]|"(?:\\.|[^"\\])*")*\}')
        matches = json_pattern.findall(text)
        for match in matches:
            try:
                parsed = json.loads(match)
                if '一致性' in parsed:
                    logging.info(f"Successfully extracted using method 2: {parsed}")
                    return parsed
            except:
                continue
    except Exception as e:
        logging.warning(f"Method 2 extraction failed: {e}")
    
    # Method 3: Try to load the entire text directly
    try:
        parsed = json.loads(text.strip())
        logging.info(f"Successfully extracted using method 3: {parsed}")
        return parsed
    except Exception as e:
        logging.warning(f"Method 3 extraction failed: {e}")
        
    logging.error("All JSON extraction methods failed")
    return None

def score_with_prompt(client, model, prompt_text, audio_path, max_retries=5):
    """Score audio using Gemini API"""
    safety = [
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    ]

    prompt_summary = prompt_text[:100] + "..." if len(prompt_text) > 100 else prompt_text
    logging.info(f"Preparing to process audio: {audio_path}, prompt text: {prompt_summary}")
    
    for retry in range(max_retries):
        uploaded = None
        try:
            logging.info(f"Processing audio: {audio_path}, attempt {retry+1}/{max_retries}")
            
            # Upload audio
            uploaded = client.files.upload(file=audio_path)
            resp = client.models.generate_content(
                model=model,
                contents=[prompt_text, uploaded],
                config=types.GenerateContentConfig(
                    response_mime_type="text/plain",
                    temperature=0,
                    safety_settings=safety
                )
            )

            text = resp.text.strip()
            usage = resp.usage_metadata
            
            if retry != 0:
                logging.info(f"Attempt {retry+1} response content:\n{text}")
                print(f"Attempt {retry+1} response content:\n{text}")
                print(f"Audio path: {audio_path}")
            
            context_info = f"[Audio: {audio_path}]"
            logging.info(f"{context_info}API returned original text: {text}")

            # Extract JSON
            result = extract_json_from_text(text)

            if result and '一致性' in result:
                logging.info(f"{context_info}Successfully extracted consistency: {result['一致性']}")
                return result['一致性'], usage
            else:
                logging.error(f"{context_info}Failed to extract consistency field from text: {text}")
                time.sleep(1)  # Brief delay before retry
                
        except Exception as e:
            logging.error(f"[Audio: {audio_path}] Error occurred during processing: {str(e)}")
            time.sleep(2)
        finally:
            if uploaded:
                try: 
                    client.files.delete(name=uploaded.name)
                    logging.info(f"Deleted uploaded file: {uploaded.name}")
                except Exception as e: 
                    logging.error(f"Failed to delete uploaded file: {str(e)}")
    
    logging.error(f"Failed to obtain scoring result after {max_retries} attempts")
    return None, None

def process_single_item(item, template, client, model, instruction_type):
    """Process a single data item"""
    result = item.copy()
    
    # Determine instruction types to process
    instruction_types = ["APS", "DSD", "RP"] if instruction_type == "ALL" else [instruction_type]
    
    for inst_type in instruction_types:
        if inst_type not in result:
            continue
            
        instruction_data = result[inst_type]
        instruction = instruction_data.get('instruction')
        audio_path = instruction_data.get('gen_path')
        
        if not audio_path or not os.path.isfile(audio_path):
            logging.warning(f"Audio file does not exist: {audio_path}")
            result[inst_type]['gemini_score'] = None
            continue
            
        if not instruction:
            logging.warning(f"Instruction is empty: {inst_type}")
            result[inst_type]['gemini_score'] = None
            continue
            
        logging.info(f"Processing {inst_type} type, instruction: {instruction[:50]}...")
        prompt_text = template.replace('<此处插入待评测的语音风格描述>', instruction)
        
        score, usage = score_with_prompt(client, model, prompt_text, audio_path)
        logging.info(f"Obtained scoring result: {score}")
        
        result[inst_type]['gemini_score'] = score
    
    return result

def worker_process(worker_id, items, template, api_key, model_name, instruction_type, result_queue):
    """Worker process function"""
    try:
        client = genai.Client(api_key=api_key)
        logging.info(f"Worker process {worker_id} started")
        
        for item in items:
            try:
                result = process_single_item(item, template, client, model_name, instruction_type)
                result_queue.put((True, result))
            except Exception as e:
                logging.error(f"Worker process {worker_id} failed to process item: {str(e)}")
                result_queue.put((False, item))
                
    except Exception as e:
        logging.error(f"Worker process {worker_id} initialization failed: {str(e)}")

def calculate_statistics(results):
    """Calculate statistics for APS, DSD, RP scores"""
    stats = {
        'APS': {'total': 0, 'true_count': 0, 'null_count': 0},
        'DSD': {'total': 0, 'true_count': 0, 'null_count': 0},
        'RP': {'total': 0, 'true_count': 0, 'null_count': 0}
    }
    
    for result in results:
        for inst_type in ['APS', 'DSD', 'RP']:
            if inst_type in result:
                stats[inst_type]['total'] += 1
                score = result[inst_type].get('gemini_score')
                if score is True:
                    stats[inst_type]['true_count'] += 1
                elif score is None:
                    stats[inst_type]['null_count'] += 1
    
    # Calculate percentages
    for inst_type in stats:
        total = stats[inst_type]['total']
        true_count = stats[inst_type]['true_count']
        null_count = stats[inst_type]['null_count']
        
        if total > 0:
            valid_count = total - null_count
            if valid_count > 0:
                percentage = (true_count / valid_count) * 100
            else:
                percentage = 0.0
        else:
            percentage = 0.0
            
        stats[inst_type]['percentage'] = percentage
        stats[inst_type]['valid_count'] = total - null_count
    
    return stats

def print_statistics(stats):
    """Print formatted statistics"""
    print("\n" + "="*60)
    print("EVALUATION STATISTICS")
    print("="*60)
    
    for inst_type in ['APS', 'DSD', 'RP']:
        data = stats[inst_type]
        print(f"{inst_type:>8}: {data['percentage']:6.2f}% ({data['true_count']:3d}/{data['valid_count']:3d} valid, {data['null_count']:3d} null)")
    
    # Calculate macro average (simple average of percentages)
    valid_types = [inst_type for inst_type in ['APS', 'DSD', 'RP'] if stats[inst_type]['valid_count'] > 0]
    if valid_types:
        macro_avg_percentage = sum(stats[inst_type]['percentage'] for inst_type in valid_types) / len(valid_types)
        print(f"{'AVG':>8}: {macro_avg_percentage:6.2f}% (macro average)")
    
    print("="*60)
    
    # Also log the statistics
    logging.info("Evaluation Statistics:")
    for inst_type in ['APS', 'DSD', 'RP']:
        data = stats[inst_type]
        logging.info(f"{inst_type}: {data['percentage']:.2f}% ({data['true_count']}/{data['valid_count']} valid, {data['null_count']} null)")
    
    if valid_types:
        logging.info(f"AVG: {macro_avg_percentage:.2f}% (macro average)")

def main():
    args = parse_args()
    logging.info(f"Starting scoring program with parameters: {vars(args)}")

    # Load prompt template
    template = load_template(args.prompt_file)
    logging.info(f"Loaded prompt template, length: {len(template)} characters")

    # Read input data
    with open(args.input_jsonl, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    
    logging.info(f"Read {len(lines)} records")

    # Parse all items
    items = [json.loads(line) for line in lines]
    logging.info(f"Parsed {len(items)} items")

    # Multi-process processing
    if args.num_workers > 1 and len(items) > 1:
        # Split data into chunks
        chunk_size = max(1, len(items) // args.num_workers)
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        # Create process pool
        manager = mp.Manager()
        result_queue = manager.Queue()
        processes = []
        
        for i, chunk in enumerate(chunks):
            p = mp.Process(
                target=worker_process,
                args=(i, chunk, template, args.api_key, args.model_name, args.instruction_type, result_queue)
            )
            p.start()
            processes.append(p)
        
        # Collect results
        results = []
        for _ in tqdm(range(len(items)), desc="Collecting results"):
            success, result = result_queue.get()
            results.append(result)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
            
    else:
        # Single process processing
        client = genai.Client(api_key=args.api_key)
        results = []
        
        for item in tqdm(items, desc="Scoring"):
            try:
                result = process_single_item(item, template, client, args.model_name, args.instruction_type)
                results.append(result)
            except Exception as e:
                logging.error(f"Failed to process item: {str(e)}")
                results.append(item)
    
    # Save results
    with open(args.output_jsonl, 'w', encoding='utf-8', buffering=1) as fout:
        for result in results:
            fout.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logging.info(f"Processing completed, results saved to: {args.output_jsonl}")
    logging.info(f"Total processed {len(results)} samples")
    
    # Calculate and print statistics
    stats = calculate_statistics(results)
    print_statistics(stats)

if __name__ == '__main__':
    main() 