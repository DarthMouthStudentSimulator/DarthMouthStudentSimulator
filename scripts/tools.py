import os
import sys
import pandas as pd
from typing import Dict, List, Tuple, Optional
import ast
import re
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import LLMClient
from models.model import EmotionStatus, AcademicScore


class BigFiveAnalyzer:
    """Calculate Bigfive score from Student survey form"""
    @staticmethod
    def compute_bigfive_scores(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path, comment='/').dropna()
        likert_map = {
            "Disagree Strongly": 1,
            "Disagree a little": 2,
            "Neither agree nor disagree": 3,
            "Agree a little": 4,
            "Agree strongly": 5
        }
        
        scored_df = df.copy()
        scored_df.iloc[:, 2:] = scored_df.iloc[:, 2:].replace(likert_map)
       
        
        reverse_items = {6, 21, 31, 2, 12, 27, 37, 8, 18, 23, 43, 9, 24, 34, 35, 41} #List of reverse score question
        question_columns = scored_df.columns[2:]
        col_number_map = {col: int(col.split("-")[1].split(".")[0].strip()) for col in question_columns}
        
        for col, q_num in col_number_map.items():
            if q_num in reverse_items:
                scored_df[col] = scored_df[col].apply(lambda x: 6 - x if pd.notnull(x) else x)
        traits = {   # Indicate relevant question with Big Five dimension
            'Extraversion': [1, 6, 11, 16, 21, 26, 31, 36],
            'Agreeableness': [2, 7, 12, 17, 22, 27, 32, 37, 42],
            'Conscientiousness': [3, 8, 13, 18, 23, 28, 33, 38, 43],
            'Neuroticism': [4, 9, 14, 19, 24, 29, 34, 39],
            'Openness': [5, 10, 15, 20, 25, 30, 35, 40, 41, 44],
        }
        
        trait_scores = []
        for _, row in scored_df.iterrows():
            scores = {'uid': row['uid'], 'type': row['type']}
            for trait, items in traits.items():
                cols = [col for col, num in col_number_map.items() if num in items]
                scores[trait] = row[cols].mean() * 20
            trait_scores.append(scores)
        
        return pd.DataFrame(trait_scores)
    

    @staticmethod
    def get_bigfive_dict(trait_df: pd.DataFrame, uid: str, type_: str) -> Dict[str, float]:
        row = trait_df[(trait_df['uid'] == uid) & (trait_df['type'] == type_)]
        if row.empty:
            raise ValueError(f"No Big Five data for uid={uid}, type={type_}")
        
        return {
            'O': row['Openness'].values[0],
            'C': row['Conscientiousness'].values[0],
            'E': row['Extraversion'].values[0],
            'A': row['Agreeableness'].values[0],
            'N': row['Neuroticism'].values[0]
        }
    

    @staticmethod
    def compute_bigfive_scores_from_answers(answers: Dict[str, int]) -> Dict[str, Optional[float]]:
        """
        Compute Big Five scores from in-memory Likert responses.
        Expects answers: {question_key: numeric_score (1-5)}.
        """
        if not answers:
            return {
                'Extraversion': None,
                'Agreeableness': None,
                'Conscientiousness': None,
                'Neuroticism': None,
                'Openness': None,
            }

        # Parse question numbers
        col_number_map = {
            col: int(col.split("-")[1].split(".")[0].strip())
            for col in answers.keys()
        }

        reverse_items = {6, 21, 31, 2, 12, 27, 37, 8, 18, 23, 43, 9, 24, 34, 35, 41}
        traits = {
            'Extraversion': [1, 6, 11, 16, 21, 26, 31, 36],
            'Agreeableness': [2, 7, 12, 17, 22, 27, 32, 37, 42],
            'Conscientiousness': [3, 8, 13, 18, 23, 28, 33, 38, 43],
            'Neuroticism': [4, 9, 14, 19, 24, 29, 34, 39],
            'Openness': [5, 10, 15, 20, 25, 30, 35, 40, 41, 44],
        }

        # Reverse-code if needed
        adjusted = {}
        for col, score in answers.items():
            qnum = col_number_map[col]
            adjusted[col] = 6 - score if qnum in reverse_items else score

        # Compute trait scores (scale to 100)
        scores = {}
        for trait, items in traits.items():
            cols = [col for col, num in col_number_map.items() if num in items]
            vals = [adjusted[col] for col in cols if adjusted[col] is not None]
            scores[trait] = sum(vals) / len(vals) * 20 if vals else None

        return scores


    @staticmethod
    def simulate_agent_likert_responses(
        llm_client,
        trait_df: pd.DataFrame,
        uid: str,
        questions: list,
        *,
        likert_map=None,
        max_retries=5,
        setup: str,
        summarize_journal_text=None) -> dict:
        """
        Simulate agent Likert responses for Big Five questions using LLM,
        then compute and return the Big Five scores as a JSON object.

        Args:
            llm_client: LLMClient instance
            trait_df: DataFrame with Big Five scores
            uid: user ID to simulate
            questions: list of question strings
            likert_map: list of Likert options (default provided)
            max_retries: max retries for invalid responses

        Returns:
            dict: Big Five scores for the agent
        """
        if likert_map is None:
            likert_map = [
                "Disagree Strongly",
                "Disagree a little",
                "Neither agree nor disagree",
                "Agree a little",
                "Agree strongly"
            ]
        likert_map_lower = {option.lower(): option for option in likert_map}
        likert_score_map = {
            "Disagree Strongly": 1,
            "Disagree a little": 2,
            "Neither agree nor disagree": 3,
            "Agree a little": 4,
            "Agree strongly": 5
        }
        def match_likert_response(answer):
            answer_lower = answer.lower()
            answer_clean = answer_lower.strip().rstrip('.').lower()
            return likert_map_lower.get(answer_clean, None)

        # Simulate agent responses
        agent_row = {"uid": uid, "type": "agent"}
        for q in questions:
            valid_response = False
            retries = 0
            while not valid_response and retries < max_retries:
                retries += 1
                system_prompt = f"""You are a university student.
You will answer Big Five Personality Test questions based on your given personality profile.

Your Personality Profile:
- Openness: f"{trait_df.loc[(trait_df['uid'] == uid) & (trait_df['type'] == 'pre'), 'Openness'].values[0]:.1f}"
- Conscientiousness: f"{trait_df.loc[(trait_df['uid'] == uid) & (trait_df['type'] == 'pre'), 'Conscientiousness'].values[0]:.1f}"
- Extraversion: f"{trait_df.loc[(trait_df['uid'] == uid) & (trait_df['type'] == 'pre'), 'Extraversion'].values[0]:.1f}"
- Agreeableness: f"{trait_df.loc[(trait_df['uid'] == uid) & (trait_df['type'] == 'pre'), 'Agreeableness'].values[0]:.1f}"
- Neuroticism: f"{trait_df.loc[(trait_df['uid'] == uid) & (trait_df['type'] == 'pre'), 'Neuroticism'].values[0]:.1f}"

Instructions:
1. Answer each question as if you are a student with these exact personality traits.
2. Be consistent with your personality profile across all questions.
3. Choose from these exact options: {', '.join(likert_map)}.
4. Return ONLY the chosen option text, nothing else.

Remember: Answer authentically based on your personality profile, not what you think is "correct".

"""
                if summarize_journal_text:
                    system_prompt += f"\nYour Recent Memories:\n{summarize_journal_text}\n"
                response = llm_client.generate(prompt ="Question : " + q,system_prompt = system_prompt)
                try:
                    if isinstance(response, dict) and "choices" in response:
                        answer = response["choices"][0]["message"]["content"].strip()
                    else:
                        answer = response.strip()
                    canonical_answer = match_likert_response(answer)
                    if canonical_answer:
                        agent_row[q] = canonical_answer
                        valid_response = True
                    else:
                        print(f"[Retry {retries}] Invalid response: '{answer}' → retrying...")
                except Exception as e:
                    print(f"[Retry {retries}] Error: {e} → retrying...")
            if not valid_response:
                print(f"[FAILED] Could not get valid answer for '{q}' after {max_retries} tries.")
                agent_row[q] = None

        # Convert Likert responses to scores
        scored_answers: Dict[str, Optional[int]] = {}
        for q in questions:
            choice = agent_row.get(q)
            scored_answers[q] = likert_score_map.get(choice) if choice is not None else None

        # Remove unanswered items before scoring
        scored_payload: Dict[str, int] = {q: v for q, v in scored_answers.items() if v is not None}

        # --- Delegate ALL scoring (incl. reverse-keying, scaling) to your scorer ---
        # Adjust this call to match your scorer's signature if needed.
        bigfive = BigFiveAnalyzer.compute_bigfive_scores_from_answers(scored_payload)

        # Ensure return shape is consistent with your previous API
        result = {'uid': uid, 'type': setup}
        result.update(bigfive)
        return result
    @staticmethod
    def summarize_journal(uid_id: str, llm_client) -> Dict:
        """
        Summarize student history for prompting in Big Five simulation.
        """
        filename = f"{uid_id}_emotion_status_history.jsonl"

        try:
            with open(filename, 'r') as f:
                school_memory = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            return {
                "uid": uid_id,
                "summary": "No memory file found."
            }

        if not school_memory:
            return {
                "uid": uid_id,
                "summary": "No memory entries available."
            }

        memory_strings = []
        for memory in school_memory:
            week = memory.get("week", "?")
            day = memory.get("day", "?")
            lab_assessment = memory.get("text", "")
            daily_desc = memory.get("daily_desc", "")
            memory_strings.append(
                f"Week {week}, Day {day} — Lab: {lab_assessment}\nNote: {daily_desc}"
            )

        joined_memories = "\n\n".join(memory_strings)

    
        summary_prompt = (
        "You are a university student.\n"
        "Below are your memory entries from a university student documenting their academic journey, personal growth, and campus life.\n"
        "Analyze these memories to create a comprehensive summary that captures:\n\n"
        "1. **Academic Journey**: Course experiences, learning challenges, study habits, and intellectual development\n"
        "2. **Social Dynamics**: Friendships, relationships, campus involvement, and social adaptation\n"
        "3. **Personal Growth**: Emotional development, self-discovery, independence, and life transitions\n"
        "4. **Recurring Themes**: Patterns in behavior, thought processes, values, and priorities\n"
        "5. **Emotional Landscape**: Predominant feelings, stress points, moments of joy, and coping mechanisms\n"
        "6. **Challenges & Resilience**: Academic struggles, social difficulties, personal obstacles, and how they were addressed\n\n"
        "Focus on creating a coherent narrative that shows the student's evolution and key experiences that shaped their university years.\n\n"
        f"{joined_memories}\n\n"
        "Provide a thoughtful analysis that helps the student understand their own journey and growth patterns:\n"
    )
        summary = llm_client.generate(summary_prompt)
        result = {
            "uid": uid_id,
            "summary": summary
        }
        output_dir = "./summary_jornal"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{uid_id}.json")
        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(result, out_f, ensure_ascii=False, indent=4)
        
        summary = llm_client.generate(summary_prompt)
        return result


class StudentAgent:
    def __init__(self, big_five: Dict[str, float], class_info: Dict, llm_client: LLMClient):
        self.big_five = big_five
        self.class_info = class_info
        self.llm_client = llm_client
        self.weekly_data = {'sensing_data': []}
        self.emotion_status = EmotionStatus()
        
    def _format_class_info(self) -> str:
        day_map = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
        result_lines = []
        
        for class_name, info in self.class_info.items():
            location = info.get('location', 'Unknown')
            periods = info.get('periods', [])
            describe = info.get('describe', 'Unknown')
            
            if periods:
                times = sorted([f"{day_map.get(p['day'], 'Day' + str(p['day']))} {p['start']}-{p['end']}" for p in periods])
                schedule_str = ", ".join(times)
            else:
                schedule_str = "TBA"
            
            result_lines.append(f"- {class_name} ({describe}): {schedule_str} at {location}")
        
        return "\n".join(result_lines)
    
    def load_real_week_data(self, week_df: pd.DataFrame):
        self.weekly_data['sensing_data'] = []
        day_summary = []
        for _, row in week_df.iterrows():
            timestamp = row['times']
            activity_flag = row[' activity inference']
            location = row['location']
            location_desc = row['location_des']
            
            hour = pd.to_datetime(timestamp).strftime('%a %H:%M')
            location_str = location if pd.notna(location) else "Unknown"
            activity_str = "Yes" if activity_flag else "No"
            
            entry_str = f"{hour} | Activity: {activity_str} | Location: {location_str} | {location_desc}"
            day_summary.append(entry_str)
        
        self.weekly_data['sensing_data'].append(day_summary)
    
    def set_weekly_class_experience(self, class_experience_df: pd.DataFrame):
        """Set class experience data"""
        self.weekly_data['class_experience'] = []
        enjoyment_scale = {
            1: "felt neutral",
            2: "strongly agreed",
            3: "agreed",
            4: "disagreed",
            5: "strongly disagreed"
        }
        
        for _, row in class_experience_df.iterrows():
            course = row['course_id']
            assignment = row["Do you have an assignment (due), quizz or exam today?"]
            enjoyment_level = row["I enjoyed the class today."]
            
            line = (
                f"In class {course}:\n"
                f"→ I had {assignment} assignment(s)/quiz/exam this week.\n"
                f"→ I {enjoyment_scale.get(int(enjoyment_level), 'had unknown feelings about')} enjoying this class."
            )
            self.weekly_data['class_experience'].append(line)
    
    def generate_journal_entry(self, deadline_text: List[str] = None ,memory_context: List[str] = None) -> str:
        """Generate daily journal entry"""
        sensing_str = "\n".join([f"Day {i+1}:\n" + "\n".join(day) for i, day in enumerate(self.weekly_data['sensing_data'])])
        class_exp = "\n".join(self.weekly_data.get("class_experience", []))
        
        system_prompt = f"""You are a university student simulator.
        You will generate a self-reflection journal based on class schedule and real-world sensing data.
        Write naturally and personally, as if you were the student reflecting on your day.
        Focus only on context - DO NOT add unnecessary elements like name or date.

        Personality:
        - Openness: {self.big_five['O']:.1f}
        - Conscientiousness: {self.big_five['C']:.1f}
        - Extraversion: {self.big_five['E']:.1f}
        - Agreeableness: {self.big_five['A']:.1f}
        - Neuroticism: {self.big_five['N']:.1f}

        Enrolled Classes:
        {self._format_class_info()}

        Current Student status: 
        happy: {self.emotion_status.happy}
        sleep: {self.emotion_status.sleep}
        social: {self.emotion_status.social}
        stamina: {self.emotion_status.stamina}
        knowledge: {self.emotion_status.stress}
        knowledge: {self.emotion_status.knowledge}

        Your Class Experience Summary:
        {class_exp}"""

        user_prompt = f"""You are a university student. This is your daily activity.
        Sensing Data for Today:
        (Each entry: Timestamp | Activity | Location | Location description)
        {sensing_str}

        TASK: Reflect on your experience today in class, on campus, and in your social life. How did you feel? Any challenges? What are your goals for toomorow?"""

        if deadline_text:
            system_prompt += f"\n\nHere are the upcoming deadlines:\n" + "\n".join(deadline_text)

        if memory_context:
            system_prompt += "\n\nRecent Memories:\n" + "\n".join([json.dumps(m) for m in memory_context])

        return self.llm_client.generate(user_prompt, system_prompt)
    
    def generate_project_submission(self) -> str:
        """Generate project submission"""
        system_prompt = f"""You are a university student simulator.
        Personality:
        - Openness: {self.big_five['O']:.1f}
        - Conscientiousness: {self.big_five['C']:.1f}
        - Extraversion: {self.big_five['E']:.1f}
        - Agreeableness: {self.big_five['A']:.1f}
        - Neuroticism: {self.big_five['N']:.1f}

        Enrolled Classes:
        {self._format_class_info()}

        Current Student status: 
        happy: {self.emotion_status.happy}
        sleep: {self.emotion_status.sleep}
        social: {self.emotion_status.social}
        stamina: {self.emotion_status.stamina}
        knowledge: {self.emotion_status.stress}
        knowledge: {self.emotion_status.knowledge}
"""

        user_prompt = f"""You are a university student. This is your last week to present final project(ideas) on smartphone programming to get 30 score.

Please generate a creative and feasible mobile app project idea that demonstrates your understanding of smartphone programming concepts."""

        return self.llm_client.generate(user_prompt, system_prompt)
    
    def analyze_emotion(self, journal_text: str) -> Tuple[EmotionStatus, str]:
        """Analyze emotional state from journal entry"""
        system_prompt = f"""You are an emotional state analyzer.
        Your task is to analyze a student's daily self-reflection journal and infer their emotional state.

        You must:
        1. Output a Python dictionary with keys: ['stamina', 'knowledge', 'stress', 'happy', 'sleep', 'social']
        - Each value should be an integer between 0 and 100.

        2. Explain briefly why each emotional value was chosen.
        - Use reasoning directly from the journal text.
        - Match student words/phrases with your judgment.

        Current Student status: 
        {self.emotion_status}



        Output format:
        {{
        "stamina": value,
        "knowledge": value,
        "stress": value,
        "happy": value,
        "sleep": value,
        "social": value
        }}

        Reasoning:
        - Stamina: because the student mentioned feeling drained after class.
        - Stress: because they worried about deadlines, etc."""

        user_prompt = f"""Here is the journal entry from the student:

        {journal_text}

        Please analyze and output both the emotional dictionary and reasoning."""

        response = self.llm_client.generate(user_prompt, system_prompt)
        
        # Extract dictionary from response
        dict_match = re.search(r"\{.*?\}", response, re.DOTALL)
        if dict_match:
            try:
                emotion_dict = ast.literal_eval(dict_match.group(0))
                emotion_status = EmotionStatus(**emotion_dict)
                reasoning = response[dict_match.end():].strip()
                return emotion_status, reasoning
            except Exception as e:
                print(f"Failed to parse emotion dictionary: {e}")
        
        return self.emotion_status, "Failed to parse emotion analysis"


class AcademicEvaluator:
    """Handles academic performance evaluation"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    @staticmethod
    def extract_answer(text: str) -> Optional[str]:

        patterns = [
            r'\*\*\s*([ABCD])\b',                                 # ** B
            r'answer\s*[:\-]?\s*([ABCD])\b',                     # answer: B
            r'the answer is\s*([ABCD])\b',                      # the answer is B
            r'the correct answer is\s*([ABCD])\)?',             # the correct answer is B or B)
            r'i choose\s*(?:option\s*)?([ABCD])\b',             # I choose B
            r'option\s*([ABCD])\b',                             # option B
            r'correct\s*[:\-]?\s*([ABCD])\b',                   # correct: B
            r'it is\s*([ABCD])\b',                              # it is B
            r'my answer is\s*([ABCD])\b',                       # my answer is B
            r'choose\s*([ABCD])\b',                             # choose B
            r'final answer\s*[:\-]?\s*([ABCD])\b',              # final answer: B
            r'\b([ABCD])\b\s*is correct',                       # B is correct
            r'\b([ABCD])\b\s*\(correct\)',                      # B (correct)
            r'^[\s\n]*([ABCD])[\s\n]*$',                        # single letter in entire text
        ]
        
        combined_pattern = re.compile('|'.join(patterns), re.IGNORECASE)
        match = combined_pattern.search(text)
        
        if match:
            for group in match.groups():
                if group:
                    return group.upper()
        
        return None
    
    def evaluate_weekly_exam(self, exam_df: pd.DataFrame, week: int, emotion_status: EmotionStatus, big_five: Dict[str, float]) -> AcademicScore:
        """Evaluate weekly exam performance"""
        weekly_exam = exam_df[exam_df["week"] == week]
        total_points = 0
        correct_answers = 0
        
        profile = f"""You are a student with the following characteristics:
        - Big Five Personality: Openness={big_five['O']:.1f}, Conscientiousness={big_five['C']:.1f}, 
        Extraversion={big_five['E']:.1f}, Agreeableness={big_five['A']:.1f}, Neuroticism={big_five['N']:.1f}
        - Current Status: 
        Stamina={emotion_status.stamina}, 
        Knowledge={emotion_status.knowledge}, 
        Stress={emotion_status.stress}, 
        Happy={emotion_status.happy}, 
        Sleep={emotion_status.sleep}, 
        Social={emotion_status.social}"""

        for _, row in weekly_exam.iterrows():
            topic = row["topic"]
            question = row["question"]
            correct_answer = row["answer"]
            points_per_question = row["point"]
            

            print(topic)
            print(correct_answer)

            prompt = f"""{profile}

            You are taking a smartphone programming class exam. Here's the question:

            Topic: {topic}
            Question: {question}

            Please provide your answer as a single letter (A, B, C, or D)."""

            response = self.llm_client.generate(prompt)
            
            student_answer = self.extract_answer(response)

            print(topic)
            print("choice correct",correct_answer)
            print("response:",response)
            print("extracted:",student_answer)
            
            if student_answer == correct_answer:
                total_points += points_per_question
                correct_answers += 1
        
        max_score = weekly_exam["point"].sum()
        
        # Handle case where no exam questions exist for this week
        if len(weekly_exam) == 0:
            return AcademicScore(
                week=week,
                topic="No exam this week",
                score=0,
                max_score=0,
                correct_answers=0,
                total_questions=0
            )
        
        return AcademicScore(
            week=week,
            topic=weekly_exam["topic"].iloc[0],
            score=total_points,
            max_score=max_score,
            correct_answers=correct_answers,
            total_questions=len(weekly_exam)
        )
    
    def evaluate_project_idea(self, submission: str) -> Dict:
        """Evaluate project idea submission"""
        system_prompt = """You are an expert university instructor and judge for a smartphone programming class. 
        Your task is to evaluate student mobile app project ideas based on the following criteria.

        Evaluation Criteria (30 - Project Idea):
        - Is the idea innovative or unique compared to existing apps?
        - Does it clearly address a real problem or user need?
        - Is the idea technically feasible for implementation by a student team within one semester?
        - Is the scope appropriate (not too simple, not too ambitious)?
        - Does it demonstrate thoughtful consideration of user experience and impact?

        Instructions:
        1. Evaluate the idea out of 30 based on the above criteria.
        2. answer in from of x/30
        remember answer in number/30

        """

        

        user_prompt = f"""Student Submission:
        {submission}

        Please provide your evaluation."""

        response = self.llm_client.generate(user_prompt, system_prompt)
        
        # Extract score from response
        score_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*30', response)
        score = float(score_match.group(1)) if score_match else 0
        
        return {
            "score": score,
            "feedback": response
        }
    
    