import os
import sys
import re
from dotenv import load_dotenv
from agents.anthropic import AnthropicClient
from agents.openai import OpenAIClient
from agents import LLMClient
import argparse


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scripts.student import StudentLifePipeline

load_dotenv()
OPENAI_API = os.getenv("OPENAI_API")

def get_matched_folders(base_path: str):
    """u01-u59"""

    pattern = re.compile(r'u([0-5][0-9])$')
    matched_folders = [
        folder for folder in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, folder)) and pattern.match(folder)
    ]
    matched_folders.sort(key=lambda x: int(x[1:]))
    return matched_folders

def main():
    base_path = "./all_data"
    matched_folders = get_matched_folders(base_path)
    print("Matched folders:", matched_folders)

    for uid in matched_folders:
        print(f"\n=== Running pipeline for {uid} ===")        
        config = {
            "big_five_path": "./dataset/BigFive.csv",
            "class_csv_path": "./dataset/education/class.csv",
            "class_info_path": "./dataset/education/class_info.json",
            "exam_path": "./dataset/education/lab_assignment.csv",
            "deadline_path": "./dataset/education/deadlines.csv",
            "uid": uid,
            "client_type": "openai",  # or "anthropic" or "openai"
            "api_key": os.getenv("OPENAI_API", OPENAI_API),
            "setup" :args.setup
        }

        required_files = [
            config["big_five_path"],
            config["class_csv_path"],
            config["class_info_path"],
            config["exam_path"],
            config["deadline_path"]
        ]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print(f"Error: Missing required files: {missing_files}")
            continue  # Skip to next uid

        if config["client_type"].lower() == "openai":
            llm_client = OpenAIClient(config["api_key"],"gpt-4o-mini")

        pipeline = StudentLifePipeline(llm_client, config)

        pipeline.run_full_pipeline(config["uid"], range(1, 11),range(1,8))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", choices=["llm", "agent"], default="llm")
    args = parser.parse_args()

    main()
