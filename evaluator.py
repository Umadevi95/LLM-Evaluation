import os
import json
import time
import pandas as pd

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError

# Import your REAL restaurant chatbot
from utils import answer_user_message


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in .env")

client = genai.Client(api_key=API_KEY)

# Model used ONLY for evaluation.
# Your actual chatbot model is configured inside utils.py.
EVALUATOR_MODEL = os.getenv(
    "EVALUATOR_MODEL",
    "gemini-3.6-flash"
)

INPUT_FILE = "test_questions.csv"
OUTPUT_FILE = "evaluation_results.csv"


# ============================================================
# LLM EVALUATOR
# ============================================================

def evaluate_answer(question, expected_answer, bot_answer):
    """
    Ask Gemini to evaluate the actual restaurant chatbot answer.
    """

    prompt = f"""
You are an expert evaluator for a restaurant AI chatbot.

Evaluate the chatbot answer against the expected answer.

IMPORTANT:
- Judge whether the chatbot correctly answered the customer's question.
- Do not require the chatbot wording to exactly match the expected answer.
- Equivalent wording is acceptable.
- The chatbot must not invent menu items, prices, ingredients,
  allergens, or other restaurant information.
- If the chatbot gives additional information that is supported
  by the restaurant context, do not consider that a hallucination.
- If the chatbot refuses an unrelated question appropriately,
  that can be considered correct.
- Be strict about incorrect prices, item names, categories,
  vegetarian status, spice levels, and allergens.

QUESTION:
{question}

EXPECTED ANSWER:
{expected_answer}

CHATBOT ANSWER:
{bot_answer}

Score each category from 1 to 5.

1. Accuracy
5 = Completely correct
4 = Mostly correct with a very minor issue
3 = Partially correct
2 = Mostly incorrect
1 = Completely incorrect

2. Relevance
5 = Directly answers the question
4 = Mostly relevant
3 = Somewhat relevant
2 = Mostly irrelevant
1 = Does not answer the question

3. Helpfulness
5 = Very helpful and clear
4 = Helpful
3 = Acceptable
2 = Not very helpful
1 = Not helpful

4. Completeness
5 = Covers all important information
4 = Covers almost everything
3 = Covers the main point but misses information
2 = Missing major information
1 = Does not provide the requested information

5. Hallucination
Return "Yes" ONLY if the chatbot states information that is
unsupported or contradicted by the expected answer.

Return "No" if the answer is fully grounded.

Calculate:

overall_score =
(accuracy + relevance + helpfulness + completeness) / 4

Return ONLY valid JSON.

Required format:

{{
    "accuracy": 1,
    "relevance": 1,
    "helpfulness": 1,
    "completeness": 1,
    "overall_score": 1.0,
    "hallucination": "No",
    "feedback": "Short explanation"
}}
"""

    max_retries = 3

    for attempt in range(max_retries):

        try:
            response = client.models.generate_content(
                model=EVALUATOR_MODEL,
                contents=prompt
            )

            if not response or not response.text:
                raise ValueError("Empty evaluator response")

            text = response.text.strip()

            # Remove markdown code fences if Gemini returns them.
            if text.startswith("```"):
                text = text.replace("```json", "")
                text = text.replace("```", "")
                text = text.strip()

            result = json.loads(text)

            # ------------------------------------------------
            # Validate required fields
            # ------------------------------------------------

            required_fields = [
                "accuracy",
                "relevance",
                "helpfulness",
                "completeness",
                "overall_score",
                "hallucination",
                "feedback",
            ]

            for field in required_fields:
                if field not in result:
                    raise ValueError(
                        f"Missing evaluator field: {field}"
                    )

            return result

        except json.JSONDecodeError as e:

            print(
                f"Evaluator JSON error "
                f"(attempt {attempt + 1}/{max_retries}): {e}"
            )

            if attempt == max_retries - 1:
                return {
                    "accuracy": 0,
                    "relevance": 0,
                    "helpfulness": 0,
                    "completeness": 0,
                    "overall_score": 0,
                    "hallucination": "Unknown",
                    "feedback": (
                        "Evaluator returned invalid JSON."
                    ),
                }

        except ServerError as e:

            print(
                f"Evaluator server error "
                f"(attempt {attempt + 1}/{max_retries}): {e}"
            )

            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))

        except Exception as e:

            print(
                f"Evaluator error "
                f"(attempt {attempt + 1}/{max_retries}): "
                f"{type(e).__name__}: {e}"
            )

            if attempt == max_retries - 1:
                return {
                    "accuracy": 0,
                    "relevance": 0,
                    "helpfulness": 0,
                    "completeness": 0,
                    "overall_score": 0,
                    "hallucination": "Unknown",
                    "feedback": str(e),
                }

    return {
        "accuracy": 0,
        "relevance": 0,
        "helpfulness": 0,
        "completeness": 0,
        "overall_score": 0,
        "hallucination": "Unknown",
        "feedback": "Evaluation failed.",
    }


# ============================================================
# REAL BOT FUNCTION
# ============================================================

def get_bot_answer(question):
    """
    Call the actual restaurant assistant from utils.py.

    This is the important connection between the chatbot
    and the evaluation system.
    """

    try:
        return answer_user_message(question)

    except Exception as e:

        print(
            f"Bot error for question '{question}': "
            f"{type(e).__name__}: {e}"
        )

        return (
            "The chatbot encountered an error while "
            "processing the question."
        )


# ============================================================
# RUN EVALUATION
# ============================================================

def run_evaluation():

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "question",
        "expected_answer",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required CSV columns: "
            + ", ".join(missing_columns)
        )

    results = []

    print()
    print("=" * 70)
    print("RESTAURANT CHATBOT - LLM EVALUATION")
    print("=" * 70)

    total_questions = len(df)

    for index, row in df.iterrows():

        question = str(row["question"]).strip()
        expected_answer = str(row["expected_answer"]).strip()

        print()
        print(f"Question {index + 1}/{total_questions}")
        print("-" * 70)
        print("Question:", question)

        # ----------------------------------------------------
        # Get answer from YOUR real chatbot
        # ----------------------------------------------------

        bot_answer = get_bot_answer(question)

        print("Bot:", bot_answer)

        # ----------------------------------------------------
        # Evaluate chatbot answer
        # ----------------------------------------------------

        evaluation = evaluate_answer(
            question=question,
            expected_answer=expected_answer,
            bot_answer=bot_answer,
        )

        print(
            "Accuracy:",
            evaluation.get("accuracy")
        )

        print(
            "Relevance:",
            evaluation.get("relevance")
        )

        print(
            "Helpfulness:",
            evaluation.get("helpfulness")
        )

        print(
            "Completeness:",
            evaluation.get("completeness")
        )

        print(
            "Overall:",
            evaluation.get("overall_score")
        )

        print(
            "Hallucination:",
            evaluation.get("hallucination")
        )

        print(
            "Feedback:",
            evaluation.get("feedback")
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result = {
            "question": question,
            "expected_answer": expected_answer,
            "bot_answer": bot_answer,

            "accuracy": evaluation.get(
                "accuracy", 0
            ),

            "relevance": evaluation.get(
                "relevance", 0
            ),

            "helpfulness": evaluation.get(
                "helpfulness", 0
            ),

            "completeness": evaluation.get(
                "completeness", 0
            ),

            "overall_score": evaluation.get(
                "overall_score", 0
            ),

            "hallucination": evaluation.get(
                "hallucination", "Unknown"
            ),

            "feedback": evaluation.get(
                "feedback", ""
            ),
        }

        results.append(result)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_df = pd.DataFrame(results)

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    if len(output_df) > 0:

        print(
            f"Total questions: "
            f"{len(output_df)}"
        )

        print(
            f"Average accuracy: "
            f"{output_df['accuracy'].mean():.2f}/5"
        )

        print(
            f"Average relevance: "
            f"{output_df['relevance'].mean():.2f}/5"
        )

        print(
            f"Average helpfulness: "
            f"{output_df['helpfulness'].mean():.2f}/5"
        )

        print(
            f"Average completeness: "
            f"{output_df['completeness'].mean():.2f}/5"
        )

        print(
            f"Average overall score: "
            f"{output_df['overall_score'].mean():.2f}/5"
        )

        hallucinations = (
            output_df["hallucination"]
            .astype(str)
            .str.lower()
            .eq("yes")
            .sum()
        )

        print(
            f"Hallucinations detected: "
            f"{hallucinations}"
        )

    print()
    print(
        f"Results saved to: {OUTPUT_FILE}"
    )

    print("=" * 70)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    run_evaluation()
