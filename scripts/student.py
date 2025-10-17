"""
Student Life Pipeline - Main pipeline class
"""
import os
import sys
import re
import glob
import pandas as pd
import csv
import json
from typing import Dict, List, Tuple


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.simple_memory import EpisodicMemory
from agents import LLMClient
from scripts.tools import BigFiveAnalyzer, AcademicEvaluator, StudentAgent
from models.model import EmotionStatus, AcademicScore


class StudentLifePipeline:
    """Main pipeline class that orchestrates the entire analysis"""
    
    def __init__(self, llm_client: LLMClient, config: Dict):
        self.llm_client = llm_client
        self.config = config
        self.big_five_analyzer = BigFiveAnalyzer()
        self.academic_evaluator = AcademicEvaluator(llm_client)
        self._load_data()
    
    def _load_data(self):
        """Load all necessary data files"""
        self.trait_df = self.big_five_analyzer.compute_bigfive_scores(self.config["big_five_path"])
        
        self.class_df = self._load_class_data(self.config["class_csv_path"])
        with open(self.config["class_info_path"], 'r') as f:
            self.class_info = json.load(f)
        
   
        self.exam_df = pd.read_csv(self.config["exam_path"])
        

        self.deadline_df = pd.read_csv(self.config["deadline_path"])
    
    def _load_class_data(self, csv_path: str) -> pd.DataFrame:
        """Load and process class data"""
        rows = []
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > 1:
                    rows.append(row)
        
        max_cols = max(len(r) for r in rows)
        colnames = ["uid"] + [f"c{i}" for i in range(1, max_cols)]
        rows_padded = [r + [None] * (max_cols - len(r)) for r in rows]
        
        return pd.DataFrame(rows_padded, columns=colnames)
    
    def _get_user_class_map(self, uid: str) -> Dict:
        """Get class information for a specific user"""
        user_row = self.class_df[self.class_df["uid"] == uid]
        if user_row.empty:
            return {}
        
      
        if len(user_row.columns) < 2:
            return {}
        
        class_list = user_row.iloc[0, 1:].dropna().tolist()
        return {
            cls: self.class_info.get(cls, {"location": "unknown", "schedule": "TBA", "instructor": "Unknown"})
            for cls in class_list
        }
    
    def _get_deadline_dates_and_counts(self, uid: str) -> List[Tuple[str, int]]:
        """Get deadline information for a specific user"""
        row = self.deadline_df[self.deadline_df['uid'] == uid]
        if row.empty:
            return []
        
   
        if len(row.columns) < 2:
            return []
        
        date_series = row.drop(columns=['uid']).iloc[0]
        valid_dates = date_series[date_series > 0]
        return list(valid_dates.items())
    
    def _format_deadline(self, deadline_data: List[Tuple[str, int]]) -> List[str]:
        """Format deadline data for display"""
        formatted = []
        for date_str, count in deadline_data:
            dt = pd.to_datetime(date_str + ' 23:59')
            formatted.append(f"{dt.strftime('%a %H:%M')} (count: {int(count)})")
        return formatted
    
    def process_student_day(
    self,
    uid: str,
    week_num: float,
    day_num: float,
    day_data_path: str,
    class_data_path: str = None,
) -> Dict:
        """Process a single day of student data (within a given week)."""
        print(f"\n=== 🗂 Processing Week {week_num} | Day {day_num} | File: {os.path.basename(day_data_path)} ===")

        # ---------- Load inputs ----------
        try:
            day_df = pd.read_csv(day_data_path)
            print(f"📊 Loaded day data with {len(day_df)} rows")

            big_five = self.big_five_analyzer.get_bigfive_dict(self.trait_df, uid=uid, type_='pre')
            print(f"👤 Loaded Big Five data for {uid}")

            class_info = self._get_user_class_map(uid)
            print(f"📚 Found {len(class_info)} classes for {uid}")

            deadline_data = self._get_deadline_dates_and_counts(uid)
            deadline_text = self._format_deadline(deadline_data)
            print(f"📅 Found {len(deadline_data)} deadlines for {uid}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise

        # ---------- Build student agent & restore last known emotion ----------
        student = StudentAgent(big_five, class_info, self.llm_client)

        emotion_history_path = f"{uid}_emotion_status_history.jsonl"
        if os.path.exists(emotion_history_path):
            with open(emotion_history_path, "r") as f:
                lines = f.readlines()
                if lines:
                    latest_status = json.loads(lines[-1])["emotion"]
                    student.emotion_status = EmotionStatus(**latest_status)

        # ---------- Load day data into student (with safe fallbacks) ----------
        if hasattr(student, "load_real_day_data"):
            student.load_real_day_data(day_df)
        else:
            # Backward-compatible: reuse weekly loader if day-specific one doesn't exist
            print("ℹ️ `load_real_day_data` not found. Falling back to `load_real_week_data` with day dataframe.")
            student.load_real_week_data(day_df)

        # ---------- Attach class experience for the specific day ----------
        if class_data_path and os.path.exists(class_data_path):
            class_df = pd.read_csv(class_data_path)
            if hasattr(student, "set_daily_class_experience"):
                student.set_daily_class_experience(class_df)
            else:
                print("ℹ️ `set_daily_class_experience` not found. Falling back to `set_weekly_class_experience`.")
                if hasattr(student, "set_weekly_class_experience"):
                    student.set_weekly_class_experience(class_df)
                else:
                    # Last-resort assignment to a generic container
                    container_key = "daily_data" if hasattr(student, "daily_data") else "weekly_data"
                    print(f"ℹ️ No setter for class experience found. Storing in `{container_key}['class_experience']`.")
                    if not hasattr(student, container_key):
                        setattr(student, container_key, {})
                    getattr(student, container_key)["class_experience"] = class_df.to_dict(orient="records")
        else:
            print(f"⚠️ No class experience data for week {week_num}, day {day_num}. Filling with blank.")
            container_key = "daily_data" if hasattr(student, "daily_data") else "weekly_data"
            if not hasattr(student, container_key):
                setattr(student, container_key, {})
            getattr(student, container_key)["class_experience"] = ["(No class experience recorded for this day.)"]

        # ---------- Journal & emotion analysis ----------
        if self.config.get("setup") == "agent":
            memory = EpisodicMemory(uid)
            recent_memories = memory.retrieve_recent(week=week_num,day=day_num)
            journal = student.generate_journal_entry(deadline_text,recent_memories)
            
        
        if self.config.get("setup") == "llm":
            journal = student.generate_journal_entry(deadline_text)
           
        emotion_status, reasoning = student.analyze_emotion(journal)
        # ---------- Academic evaluation (day-aware if available) ----------
        try:
            if hasattr(self.academic_evaluator, "evaluate_daily_exam"):
                academic_score = self.academic_evaluator.evaluate_daily_exam(
                    self.exam_df, week_num, day_num, emotion_status, big_five
                )
            else:
                # Backward-compatible: use weekly evaluation
                academic_score = self.academic_evaluator.evaluate_weekly_exam(
                    self.exam_df, int(week_num), emotion_status, big_five
                )
            print(f"📝 Academic evaluation completed for week {week_num}, day {day_num}")
        except Exception as e:
            print(f"❌ Error in academic evaluation: {e}")
            academic_score = AcademicScore(
                week=int(week_num),
                topic="Error in evaluation",
                score=0,
                max_score=0,
                correct_answers=0,
                total_questions=0
            )

        # ---------- Optional project checkpoint (kept compatible with old logic) ----------
        if int(week_num) == 10 and hasattr(student, "generate_project_submission"):
            idea_submission = student.generate_project_submission()
            project_result = self.academic_evaluator.evaluate_project_idea(submission=idea_submission)
            result = {
                "week": week_num,
                "day": day_num,
                "emotion": emotion_status.__dict__,
                "lab_assessment": {
                    "score": academic_score.score,
                    "max_score": academic_score.max_score,
                    "topic": academic_score.topic,
                    "correct_answers": academic_score.correct_answers,
                    "total_questions": academic_score.total_questions,
                    "week": academic_score.week,
                },
                "project": {
                    "score": project_result["score"],
                    "full_text_response": project_result["feedback"],
                },
                "daily_desc": journal,
                "judge_reasoning": reasoning,
            }
        else:
            result = {
                "week": week_num,
                "day": day_num,
                "emotion": emotion_status.__dict__,
                "lab_assessment": {
                    "score": academic_score.score,
                    "max_score": academic_score.max_score,
                    "topic": academic_score.topic,
                    "correct_answers": academic_score.correct_answers,
                    "total_questions": academic_score.total_questions,
                    "week": academic_score.week,
                },
                "daily_desc": journal,
                "judge_reasoning": reasoning,
            }

        # ---------- Persist results ----------
        with open(emotion_history_path, "a") as f:
            json.dump(result, f)
            f.write("\n")

        with open(f"{uid}_emotion_status.json", "w") as f:
            json.dump(emotion_status.__dict__, f)

        return result
        
    
    
    
    
    def run_full_pipeline(self, uid: str, weeks_range: range = range(1, 11), days_range: range = range(1, 8)) -> List[Dict]:
        """Run the complete pipeline for multiple weeks and days"""
        results: List[Dict] = []

        base_dir = f"./all_data/{uid}"
        os.makedirs("./student_status", exist_ok=True)

        # Load Big Five questions once
        bf_df = pd.read_csv(self.config["big_five_path"])
        question_list = bf_df.columns.drop(['uid', 'type']).tolist()

        for week in weeks_range:
            week_f = float(f"{week:.1f}")  # keep float-like formatting for filenames

            print(f"\n================= 📆 Starting Week {week} =================")
            # Process each day in this week
            for day in days_range:
                day_f = float(f"{day:.1f}")

                data_path = os.path.join(base_dir, f"data_week{week_f:.1f}_day{day_f:.1f}.csv")
                class_path = os.path.join(base_dir, f"class_week{week_f:.1f}_day{day_f:.1f}.csv")

                if not os.path.exists(data_path):
                    print(f"⚠️  Missing data file: {os.path.basename(data_path)} — skipping.")
                    continue

                class_data_path = class_path if os.path.exists(class_path) else None
                if class_data_path is None:
                    print(f"ℹ️  No class file for day {day} (week {week}).")

                # Day-level processing
                try:
                    result = self.process_student_day(uid, week_f, day_f, data_path, class_data_path)
                    results.append(result)
                except Exception as e:
                    print(f"❌ Error processing week {week} day {day}: {e}")

        # ===== After finishing all days in the current week → run Big Five once =====
        print("-------------------- Simulation agent doing Big5 (end of week) -------------------")
        if self.config.get("setup") == "agent":
            
            
            summary_text  = BigFiveAnalyzer.summarize_journal(uid_id  = uid,llm_client= self.llm_client)
           
            simulated_scores = BigFiveAnalyzer.simulate_agent_likert_responses(
                    llm_client=self.llm_client,
                    trait_df=self.trait_df,
                    uid=uid,
                    questions=question_list,
                    setup = self.config.get("setup"),
                    summarize_journal_text = summary_text 

                )
        if self.config.get("setup") == "llm":
         simulated_scores = BigFiveAnalyzer.simulate_agent_likert_responses(
                llm_client=self.llm_client,
                trait_df=self.trait_df,
                uid=uid,
                questions=question_list,
                setup = self.config.get("setup")
            )

            # Save per-week Big Five simulation to avoid overwriting
        os.makedirs("./student_status/{uid}", exist_ok=True)
        output_path = f"./student_status/{self.config.get('setup')}_{uid}_simulated_agent_big5.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(simulated_scores, f, ensure_ascii=False, indent=4)

        print(f"✅ Big5 simulation completed for week {week}. JSON saved to: {output_path}")
        print(f"================= ✅ Finished Week {week} =================\n")

        return results
    
    def evaluate_project_submission(self, submission: str) -> Dict:
        """Evaluate a project idea submission"""
        return self.academic_evaluator.evaluate_project_idea(submission)