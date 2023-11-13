example1 = """
def encrypt(s):
    '''
    Create a function encrypt that takes a string as an argument and returns a string encrypted with the alphabet being
    rotated. The alphabet should be rotated in a manner such that the letters shift down by two multiplied to two places.
    For example:
    encrypt(’hi’) returns ’lm’
    encrypt(’asdfghjkl’) returns ’ewhjklnop’
    encrypt(’gf’) returns ’kj’
    encrypt(’et’) returns ’ix’
    Let’s think step by step.
    1. Create a alphabet, bias two places multiplied by two.
    2. Loop the input, find the latter bias letter in alphabet.
    3. Return result.
    '''
"""

example2 = """
# This is an example how to write code with plan
def encrypt(s):
    '''QUESTION
    Create a function encrypt that takes a string as an argument and returns a string encrypted with the alphabet being
    rotated. The alphabet should be rotated in a manner such that the letters shift down by two multiplied to two places.
    For example:
    encrypt(’hi’) returns ’lm’
    encrypt(’asdfghjkl’) returns ’ewhjklnop’
    encrypt(’gf’) returns ’kj’
    encrypt(’et’) returns ’ix’
    '''
    
    '''Give your plan
    Let’s think step by step.
    1. Create a alphabet, bias two places multiplied by two.
    2. Loop the input, find the latter bias letter in alphabet.
    3. Return result.
    '''
    
    # Write your code here.
    alphabet = ’abcdefghijklmnopqrstuvwxyz’
    bias = 2 * 2
    result = ”
    for char in s:
    index = alphabet.find(char) + bias
    if index > 25:
    index = index - 26
    result += alphabet[index]
    return result
    
# Please follow the example above to solve the following problem with code.
"""

example3 = """
# This is an example how to write code with plan
def encrypt(s):
    '''QUESTION
    Create a function encrypt that takes a string as an argument and returns a string encrypted with the alphabet being
    rotated. The alphabet should be rotated in a manner such that the letters shift down by two multiplied to two places.
    For example:
    encrypt(’hi’) returns ’lm’
    encrypt(’asdfghjkl’) returns ’ewhjklnop’
    encrypt(’gf’) returns ’kj’
    encrypt(’et’) returns ’ix’
    '''

    '''Give your plan
    Let’s think step by step.
    1. Create a alphabet, bias two places multiplied by two.
    2. Loop the input, find the latter bias letter in alphabet.
    3. Return result.
    '''

    # Write your code here.
    alphabet = ’abcdefghijklmnopqrstuvwxyz’
    bias = 2 * 2
    result = ”
    for char in s:
    index = alphabet.find(char) + bias
    if index > 25:
    index = index - 26
    result += alphabet[index]
    return result

# Please follow the example above to solve the following problem with code.Just follow the example and don't create anything extra that affects the running of the code.
"""


example4_plan = """
# This are examples how to prepare a plan to write code in the future. 
# Example 1
def encrypt(s):
    ```
    Create a function encrypt that takes a string as an argument and returns a string encrypted with the alphabet being
    rotated. The alphabet should be rotated in a manner such that the letters shift down by two multiplied to two places.
    For example:
    encrypt(’hi’) returns ’lm’
    encrypt(’asdfghjkl’) returns ’ewhjklnop’
    encrypt(’gf’) returns ’kj’
    encrypt(’et’) returns ’ix’
    Let’s think step by step.
    1. Create a alphabet, bias two places multiplied by two.
    2. Loop the input, find the latter bias letter in alphabet.
    3. Return result.
    ```
# Example 2
def check_if_last_char_is_a_letter(txt):
    ```
    Create a function that returns True if the last character of a given string is an alphabetical character and is not a
    part of a word, and False otherwise. Note: ”word” is a group of characters separated by space.
    Examples:
    check if last char is a letter(”apple pie”) → False
    check if last char is a letter(”apple pi e”) → True
    check if last char is a letter(”apple pi e ”) → False
    check if last char is a letter(””) → False
    Let’s think step by step.
    1. If the string is empty, return False.
    2. If the string does not end with a alphabetical character, return False.
    3. Split the given string into a list of words.
    4. Check if the length of the last word is equal to 1.
    ```
# Example 3    
def file_name_check(file name):
    ```
    Create a function which takes a string representing a file’s name, and returns ’Yes’ if the the file’s name is valid,
    and returns ’No’ otherwise. A file’s name is considered to be valid if and only if all the following conditions are met:
    - There should not be more than three digits (’0’-’9’) in the file’s name. - The file’s name contains exactly one dot
    ’.’ - The substring before the dot should not be empty, and it starts with a letter from the latin alphapet (’a’-’z’ and
    ’A’-’Z’). - The substring after the dot should be one of these: [’txt’, ’exe’, ’dll’]
    Examples:
    file name check(”example.txt”) => ’Yes’
    file name check(”1example.dll”) => ’No’ (the name should start with a latin alphapet letter)
    Let’s think step by step.
    1. Check if the file name is valid according to the conditions.
    2. Return ”Yes” if valid, otherwise return ”NO”.
    ```
    
#  Please follow the example above to prepare a plan for following problem
"""



