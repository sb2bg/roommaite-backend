from faker import Faker
import random

fake = Faker()
from faker import Faker
import random
import json
from tqdm import tqdm
from datetime import date

fake = Faker()


class OpenEndedQuestion:
    def __init__(self, question_text):
        self.question_text = question_text


class YesNoQuestion:
    def __init__(self, question_text):
        self.question_text = question_text


required_questions = [
    OpenEndedQuestion("What is your name?"),
    OpenEndedQuestion("What is your age?"),
    OpenEndedQuestion("What is your education level?"),
    OpenEndedQuestion("What is your occupation?"),
    OpenEndedQuestion("What is your major?"),
    OpenEndedQuestion("What is your level of cleanliness?"),
    OpenEndedQuestion("What is your level of noise tolerance?"),
    OpenEndedQuestion("What time do you usually go to bed?"),
    OpenEndedQuestion("What time do you usually wake up?"),
    YesNoQuestion("Do you smoke?"),
    YesNoQuestion("Do you drink?"),
    OpenEndedQuestion("Do you have any pets?"),
    YesNoQuestion("Do you have any dietary restrictions?"),
    OpenEndedQuestion("What is your preferred number of roommates?"),
]

optional_questions = [
    OpenEndedQuestion("What is your budget?"),
    OpenEndedQuestion("What is your preferred move-in date?"),
    OpenEndedQuestion("What is your preferred lease length?"),
    OpenEndedQuestion("What is your preferred neighborhood?"),
]


def generate_fake_response(question):
    if isinstance(question, OpenEndedQuestion):
        if "name" in question.question_text.lower():
            return fake.name()
        elif "age" in question.question_text.lower():
            return random.randint(18, 30)
        elif "education level" in question.question_text.lower():
            return random.choice(["High School", "Undergraduate", "Graduate", "PhD"])
        elif "occupation" in question.question_text.lower():
            return fake.job()
        elif "major" in question.question_text.lower():
            return random.choice(
                ["Computer Science", "Biology", "Engineering", "Psychology", "Business"]
            )
        elif "cleanliness" in question.question_text.lower():
            return random.choice(["Very clean", "Moderately clean", "Not very clean"])
        elif "noise tolerance" in question.question_text.lower():
            return random.choice(["Low", "Moderate", "High"])
        elif "bed" in question.question_text.lower():
            return f"{random.randint(9, 12)} PM"
        elif "wake up" in question.question_text.lower():
            return f"{random.randint(6, 9)} AM"
        elif "number of roommates" in question.question_text.lower():
            return random.randint(1, 4)
        elif "budget" in question.question_text.lower():
            return f"${random.randint(500, 1500)} per month"
        elif "move-in date" in question.question_text.lower():
            return fake.date_this_year().strftime("%Y-%m-%d")
        elif "lease length" in question.question_text.lower():
            return random.choice(["6 months", "1 year"])
        elif "neighborhood" in question.question_text.lower():
            return fake.city()
        elif "pets" in question.question_text.lower():
            return random.choice(["Dog", "Cat", "Fish", "Bird", "None"])
    elif isinstance(question, YesNoQuestion):
        return random.choice(["Yes", "No"])
    return "N/A"


# Generate responses for all questions
def generate_fake_profile():
    profile = {}
    for question in required_questions + optional_questions:
        profile[question.question_text] = generate_fake_response(question)
    return profile


# Generate and save profiles to JSON

# Generate and print a sample profile
sample_profile = generate_fake_profile()
for question, response in sample_profile.items():
    print(f"{question}: {response}")
