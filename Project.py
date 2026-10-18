import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FAST_LIMIT_SEC = 60         
SLOW_LIMIT_SEC = 120         
LOW_ACCURACY = 40            
HIGH_ACCURACY = 75           
REVISION_THRESHOLD = 60      
REVIEW_ACCURACY = 60         
LEVELS = ["Easy", "Moderate", "Difficult"]


data = {
    "Question_ID": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6",
                    "Q7", "Q8", "Q9", "Q10", "Q11", "Q12"],
    "Topic": ["Python Basics"] * 3 + ["Loops"] * 3
             + ["Functions"] * 3 + ["NumPy"] * 3,
    "Marks": [2, 2, 5, 2, 5, 5, 2, 5, 5, 2, 5, 5],
    "Attempts": [100, 90, 80, 100, 85, 75,
                 90, 80, 70, 100, 85, 75],
    "Correct_Answers": [90, 65, 30, 85, 50, 25,
                        75, 40, 20, 95, 55, 25],
    "Avg_Time_Sec": [25, 55, 140, 35, 85, 160,
                     40, 100, 180, 20, 90, 150],
    "Teacher_Difficulty": [
        "Easy", "Easy", "Difficult",          
        "Easy", "Easy", "Difficult",          
        "Easy", "Difficult", "Moderate",      
        "Easy", "Moderate", "Difficult"
    ],
}

df = pd.DataFrame(data)

assert (df["Attempts"] > 0).all(), "Attempts must be greater than 0"
assert (df["Correct_Answers"] <= df["Attempts"]).all(), \
    "Correct answers cannot exceed attempts"
assert df["Teacher_Difficulty"].isin(LEVELS).all(), \
    "Teacher difficulty must be Easy, Moderate or Difficult"

def time_level(seconds):

    if seconds < FAST_LIMIT_SEC:
        return "Fast"
    elif seconds < SLOW_LIMIT_SEC:
        return "Moderate"
    return "Slow"


def classify_difficulty(accuracy, level):
    
    if accuracy < LOW_ACCURACY or level == "Slow":
        return "Difficult"
    elif accuracy >= HIGH_ACCURACY and level == "Fast":
        return "Easy"
    return "Moderate"


def review_reason(row):

    reasons = []
    if row["Difficulty_Match"] == "Mismatch":
        reasons.append(
            f"Teacher={row['Teacher_Difficulty']} vs Data={row['Data_Difficulty']}"
        )

    if (row["Correctness_Rate"] < REVIEW_ACCURACY
            and row["Teacher_Difficulty"] != "Difficult"):
        reasons.append("Low accuracy for a non-Difficult question")
    return "; ".join(reasons)


df["Correctness_Rate"] = df["Correct_Answers"] / df["Attempts"] * 100
df["Avg_Marks_Earned"] = df["Marks"] * df["Correctness_Rate"] / 100
df["Time_Level"] = df["Avg_Time_Sec"].apply(time_level)

df["Data_Difficulty"] = df.apply(
    lambda r: classify_difficulty(r["Correctness_Rate"], r["Time_Level"]),
    axis=1,
)

df["Difficulty_Match"] = np.where(
    df["Teacher_Difficulty"] == df["Data_Difficulty"], "Match", "Mismatch"
)
df["Review_Reason"] = df.apply(review_reason, axis=1)

print("\nQUESTION-LEVEL ANALYSIS")
print(df.drop(columns="Review_Reason").round(2).to_string(index=False))

topic_stats = df.groupby("Topic").agg(
    Total_Questions=("Question_ID", "count"),
    Total_Attempts=("Attempts", "sum"),
    Total_Correct=("Correct_Answers", "sum"),
    Avg_Time=("Avg_Time_Sec", "mean"),
    Avg_Marks_Earned=("Avg_Marks_Earned", "mean"),
).reset_index()

topic_stats["Overall_Correctness"] = (
    topic_stats["Total_Correct"] / topic_stats["Total_Attempts"] * 100
)
topic_stats["Topic_Time_Level"] = topic_stats["Avg_Time"].apply(time_level)
topic_stats["Topic_Difficulty"] = topic_stats.apply(
    lambda r: classify_difficulty(r["Overall_Correctness"], r["Topic_Time_Level"]),
    axis=1,
)
topic_stats["Needs_Revision"] = np.where(
    topic_stats["Overall_Correctness"] < REVISION_THRESHOLD, "Yes", "No"
)

print("\nTOPIC-WISE PERFORMANCE")
print(topic_stats.round(2).to_string(index=False))

print("\nTOPICS THAT MAY NEED REVISION")
revision_topics = topic_stats[topic_stats["Needs_Revision"] == "Yes"]
if revision_topics.empty:
    print("None")
else:
    print(
        revision_topics[["Topic", "Overall_Correctness", "Avg_Time"]]
        .round(2).to_string(index=False)
    )

review_questions = df[df["Review_Reason"] != ""]

print("\nQUESTIONS NEEDING REVIEW")
if review_questions.empty:
    print("None")
else:
    print(
        review_questions[[
            "Question_ID", "Topic", "Correctness_Rate", "Avg_Time_Sec",
            "Teacher_Difficulty", "Data_Difficulty", "Review_Reason",
        ]].round(2).to_string(index=False)
    )

comparison = pd.crosstab(
    df["Teacher_Difficulty"], df["Data_Difficulty"]
).reindex(index=LEVELS, columns=LEVELS, fill_value=0)

print("\nDIFFICULTY COMPARISON (rows = Teacher, columns = Data)")
print(comparison)

match_rate = df["Difficulty_Match"].eq("Match").mean() * 100
print(f"\nDifficulty agreement: {match_rate:.2f}%")

print("\nQUESTIONS SORTED FROM HARDEST TO EASIEST")
print(
    df.sort_values("Correctness_Rate")[[
        "Question_ID", "Topic", "Correctness_Rate",
        "Avg_Time_Sec", "Data_Difficulty",
    ]].round(2).to_string(index=False)
)

ax = topic_stats.plot(
    x="Topic", y="Overall_Correctness", kind="bar", legend=False,
    title="Topic-wise Correctness Rate", ylabel="Correctness Rate (%)",
    xlabel="Topic",
)
ax.axhline(REVISION_THRESHOLD, color="red", linestyle="--",
           label=f"Revision threshold ({REVISION_THRESHOLD}%)")
ax.legend()
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

ax = df.plot(
    x="Question_ID", y="Avg_Time_Sec", kind="bar", legend=False,
    title="Average Completion Time by Question",
    ylabel="Time (seconds)", xlabel="Question ID",
)
ax.axhline(FAST_LIMIT_SEC, color="green", linestyle="--", label="Fast limit")
ax.axhline(SLOW_LIMIT_SEC, color="red", linestyle="--", label="Slow limit")
ax.legend()
plt.tight_layout()
plt.show()