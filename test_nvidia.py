from openai import OpenAI
import json
import sys
import os

# CLEAN VERSION: No hardcoded keys
# Run with: NVIDIA_API_KEY=your_key python test_nvidia.py
API_KEY = os.getenv("NVIDIA_API_KEY")

if not API_KEY:
    print("ERROR: Please set the NVIDIA_API_KEY environment variable.")
    sys.exit(1)

print("--- STARTING NVIDIA API TEST ---")
try:
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=API_KEY)
    
    MODEL = "deepseek-ai/deepseek-r1-distill-qwen-32b"
    
    print(f"Testing Chat Completion with model: {MODEL}")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a quantitative data analyst."},
            {"role": "user", "content": "Explain briefly: What is a bullish crossover?"}
        ],
        max_tokens=256,
        temperature=0.2
    )
    
    print("\nSUCCESS! Response received:")
    print(response.choices[0].message.content)

except Exception as e:
    print(f"\nFAILED: {e}")

print("\n--- TEST FINISHED ---")
