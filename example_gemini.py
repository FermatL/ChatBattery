#!/usr/bin/env python3

"""
Example script demonstrating how to use ChatBattery with Gemini models
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ChatBattery.LLM_agent import LLM_Agent

def main():
    print("ChatBattery Gemini Integration Example")
    print("=====================================")
    
    # Check if API key is set
    if not os.getenv('GOOGLE_API_KEY'):
        print("\nTo run this example, please set your Google API key:")
        print("export GOOGLE_API_KEY=your_api_key_here")
        print("\nGet your API key from: https://ai.google.dev/")
        return
    
    # Example messages for battery optimization
    messages = [
        {"role": "system", "content": "You are an expert in the field of material and chemistry."},
        {"role": "user", "content": "We have a Li cathode material LiCoO2. Can you optimize it to develop new cathode materials with higher capacity and improved stability? Please propose three optimized battery formulations in bullet points with asterisk *."}
    ]
    
    # Available Gemini models
    gemini_models = ["gemini_1.5_flash", "gemini_1.5_pro", "gemini_1.0_pro"]
    
    print(f"\nTesting battery optimization with different Gemini models...")
    
    for model in gemini_models:
        print(f"\n--- Testing {model} ---")
        try:
            raw_text, battery_list = LLM_Agent.optimize_batteries(messages, model, temperature=0.3)
            
            print(f"Generated response:")
            print(raw_text[:300] + "..." if len(raw_text) > 300 else raw_text)
            
            print(f"\nExtracted battery formulations:")
            for i, battery in enumerate(battery_list, 1):
                print(f"  {i}. {battery}")
                
        except Exception as e:
            print(f"Error with {model}: {e}")
    
    print("\n--- Testing battery ranking functionality ---")
    ranking_messages = [
        {"role": "user", "content": "Please rank these lithium battery cathode materials by their theoretical capacity: LiCoO2, LiFePO4, LiMn2O4, LiNiO2. Explain your reasoning."}
    ]
    
    try:
        ranking_result = LLM_Agent.rank_batteries(ranking_messages, "gemini_1.5_flash", temperature=0.1)
        print(f"Ranking result:")
        print(ranking_result[:400] + "..." if len(ranking_result) > 400 else ranking_result)
        
    except Exception as e:
        print(f"Error with ranking: {e}")

if __name__ == "__main__":
    main()