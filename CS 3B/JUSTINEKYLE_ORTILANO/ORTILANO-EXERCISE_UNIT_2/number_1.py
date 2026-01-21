# ELIZA implementation in Python
# Example Generated via ChatGPT
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
        "me": "you"
    }
    words = fragment.lower().split()
    return ' '.join([reflections.get(word, word) for word in words])

def eliza_response(user_input):
    """Generates ELIZA-style responses based on input."""
    patterns = [
        (r"I need (.*)", "Why do you need {0}?"),
        (r"Why don’t you (.*)", "Do you really think I don't {0}?"),
        (r"I feel (.*)", "Tell me more about feeling {0}."),

    # Modified patterns for the exercise
        (r"I want to know the reason why I am feeling(.*)", "Why do you think you're feeling {0}"),
        (r"I am feeling(.*)", "Tell me more about what you're feeling {0}"),
        (r"My feelings(.*)", "Why do you feel that way?"),
        (r"You (don't|do not understand me)(.*)", "Why do you feel like I don't understand you?"),   
        (r"I (can't|cannot) focus on my studies", "Why do you feel like you cannot focus on your studies?"),
    ]
    
    for pattern, response in patterns:
        match = re.match(pattern, user_input, re.IGNORECASE)
       #print(match)
        if match:
            print(match.group(1)) # captures the substring after the pattern
            return response.format(reflect(match.group(1)))
    
    return "Can you tell me more?"

print("ELIZA: Hello! How can I help you today?")

#laziest way I can do to get the bonus points - just check for repeated inputs inputs in this code block here, is it effective? Probably not, but do i have to tinker with the code somewhere else NO!! (I may be confidently wrong here)
user_input = ""
while True:   
    previous_input = user_input
    user_input = input("You: ")
   

    if user_input.lower() in ["quit", "exit"]:
        print("ELIZA: Goodbye!")
        break
    if user_input.strip() == "":
        print("ELIZA: So are we going to talk or what?")
    elif user_input.strip() == previous_input.strip():
        print("ELIZA: We could go around and around all day, or do something more worth yur time.")
    else:
        print(f"ELIZA: {eliza_response(user_input)}")