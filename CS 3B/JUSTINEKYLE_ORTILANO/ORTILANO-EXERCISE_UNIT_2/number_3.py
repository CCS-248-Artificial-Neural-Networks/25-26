import re

pattern= r"\bwhale\b"

file_path = r'ORTILANO-EXERCISE_UNIT_2\\melville-moby_dick.txt' 

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
    
output = re.findall(pattern, content)

count = 0
outstring = ""
for i in output :
  count= count + 1
  outstring = outstring + i + " "

print(outstring)
print("word instances count of the word 'whale': ", count)

