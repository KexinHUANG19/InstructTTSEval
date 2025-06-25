#!/bin/bash

# ====== User Configurable Variables ======
# Path to input JSONL file
INPUT_JSONL="example_en.jsonl"
# Path to output JSONL file
OUTPUT_JSONL="example_en_score.jsonl"
# Path to prompt template
PROMPT_FILE="eval_prompt.txt"
# Gemini API key (or use environment variable)
API_KEY="$GENAI_API_KEY" 
# Gemini model name
MODEL_NAME="models/gemini-2.5-pro-preview-05-06"
# Instruction type: ALL, APS, DSD, RP
INSTRUCTION_TYPE="ALL"
# Number of worker processes
NUM_WORKERS=10

# ====== Run Evaluation ======
python gemini_eval.py \
    --input_jsonl "$INPUT_JSONL" \
    --output_jsonl "$OUTPUT_JSONL" \
    --prompt_file "$PROMPT_FILE" \
    --api_key "$API_KEY" \
    --model_name "$MODEL_NAME" \
    --instruction_type "$INSTRUCTION_TYPE" \
    --num_workers "$NUM_WORKERS"

echo "Evaluation completed! Results saved in $OUTPUT_JSONL" 