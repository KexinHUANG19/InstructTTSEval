# TTS Evaluation with Gemini API

This folder evaluates TTS-generated audio using Google's Gemini API for style consistency assessment.

## Environment Setup

### 1. Install Python Dependencies

```bash
# Install required packages
pip install google-genai requests tqdm

# Or install from requirements.txt
pip install -r requirements.txt
```

### 2. Set Up API Key

```bash
# Set your Gemini API key as environment variable
export GENAI_API_KEY="your_api_key_here"
```

## Data Format Requirements

Please prepare separate JSONL files for English (EN) and Chinese (ZH) splits. Each line in your JSONL file must follow this exact structure:

```json
{
  "id": "sample_id",
  "text": "The text content to be synthesized",
  "APS": {
    "instruction": "Detailed voice style instruction for APS type",
    "gen_path": "path/to/generated_audio_aps.wav"
  },
  "DSD": {
    "instruction": "Detailed voice style instruction for DSD type", 
    "gen_path": "path/to/generated_audio_dsd.wav"
  },
  "RP": {
    "instruction": "Detailed voice style instruction for RP type",
    "gen_path": "path/to/generated_audio_rp.wav"
  }
}
```

### Required Fields:  

- **`id`**: Unique identifier for each sample (string)
- **`text`**: The text content that was synthesized (string)
- **`APS`**: Object containing APS (Attribute-based Prompt Style) instruction and audio
  - `instruction`: Detailed voice style description
  - `gen_path`: Path to the generated audio file (relative to script location)
- **`DSD`**: Object containing DSD (Direct Style Description) instruction and audio
- **`RP`**: Object containing RP (Role-based Prompt) instruction and audio

### Audio File Requirements:

- Audio files must exist at the paths specified in `gen_path`
- Supported formats: WAV, MP3, etc. (as supported by Gemini API)
- Use relative paths from the script execution directory


## Running the Evaluation

### 1. Using the Shell Script (Recommended)

Edit `run_gemini_eval.sh` :

```bash
# ====== User Configurable Variables ======
INPUT_JSONL="example_en.jsonl"           # Input JSONL file
OUTPUT_JSONL="example_en_score.jsonl"    # Output JSONL file
PROMPT_FILE="eval_prompt.txt"         # Prompt template
API_KEY="$GENAI_API_KEY"              # Gemini API key (use env variable or set directly)
MODEL_NAME="models/gemini-2.5-pro-preview-05-06"  # Gemini model name
INSTRUCTION_TYPE="ALL"                # ALL, APS, DSD, RP
NUM_WORKERS=10                         # Number of worker processes
```

Run:

```bash
chmod +x run_gemini_eval.sh
./run_gemini_eval.sh
```

## Output Format

The evaluation script will:

1. **Process each audio file** using the corresponding instruction
2. **Call Gemini API** to assess style consistency
3. **Update `gemini_score`** fields with `true` (consistent with the given instruction) or `false` (inconsistent)
4. **Generate statistics** showing success rates for each instruction type
5. **Save results** to the output JSONL file

### Example Output Statistics:

```
============================================================
EVALUATION STATISTICS
============================================================
     APS:  88.80% ( 888/ 1000 valid,   0 null)
     DSD:  77.70% ( 777/ 1000 valid,   0 null)
      RP:  66.60% ( 666/ 1000 valid,   0 null)
     AVG:  77.70% (macro average)
============================================================
```