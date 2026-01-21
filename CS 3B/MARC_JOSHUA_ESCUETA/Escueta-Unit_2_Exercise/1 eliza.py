# ELIZA implementation in Python
import re

def reflect(fragment):
    """Reflects user input to make responses more natural."""
    reflections = {
        "am": "are",
        "was": "were",
        "i": "you",
        "i'd": "you would",
        "i've": "you have",
        "i'll": "you will",
        "my": "your",
        "are": "am",
        "you've": "I have",
        "you'll": "I will",
        "your": "my",
        "yours": "mine",
        "you": "me",
        "me": "you",
        "mama": "papa",
        "mom": "dad",
    }

    words = fragment.lower().split()
    return " ".join([reflections.get(word, word) for word in words])


def eliza_response(user_input):
    """Generates ELIZA-style responses based on input."""

    patterns = [
        (
            r".*(I want to know the reasons why I am feeling (.*) all the time).*",
            "You said that {0}. Can you tell me more about why you feel this way?"
        ),

        (
            r".*(I am feeling (.*)).*",
            "It sounds like {0}. What do you think is causing your {0}?"
        ),

        (
            r".*(My feelings towards my crush are invalidated).*",
            "You feel that {0}. How does that make you feel about yourself?"
        ),

        (
            r".*(You (?:don't|do not) understand me).*",
            "You feel that {0}. Why do you think I don't understand you?"
        ),

        (
            r".*(I (?:can't|cannot) focus on my studies).*",
            "You mentioned that {0}. What makes it hard to focus on your studies?"
        ),

        (
            r".*(apple bottom jeans boots with the (?:fur).*).*",
            "You mentioned that {0}. What makes it hard to focus on dancing?"
        ),
        
        # (
        #     r".*((?:ai) art is bad ).*",
        #     "You mentioned that {0}. What makes it bad?"
        # ),

        # """Generates ELIZA-style responses based on input."""
        (r"I need (.*)", "Why do you need {0}?"),
        (r"Why don’t you (.*)", "Do you really think I don't {0}?"),
        (r"I feel (.*)", "Tell me more about feeling {0}."),
    ]

    for pattern, response in patterns:
        match = re.match(pattern, user_input, re.IGNORECASE)
        if match:
            captured = match.group(1)
            return response.format(reflect(captured))

    return "Can you tell me more?"

print("ELIZA: Hello! How can I help you today?")
print('       (Type "quit" or "exit" to end the conversation.)')

last_input = None  # for BONUS: Detect repeated questions

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("ELIZA: Goodbye!")
        break

    normalized = user_input.strip().lower()

    if normalized == last_input:
        print("ELIZA: Asking the same thing again? "
            "Do you think repeating the question will magically change my answer?")
        continue

    last_input = normalized
    print(f"ELIZA: {eliza_response(user_input)}")
