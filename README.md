# RSP Recognition
1. To install the dependencies run `pip install -r requirements.txt` (pip3 if Mac or Linux)
2. If ran for the first time, run [`get_data.py`](trainers/get_data.py) inside the *trainers* directory
3. Then, run the *main* script and Vuala!

## How it works
When gathering data, the program measures the distance between the base of the hand to each fingertip, and that data is then submitted to the SVM and labeled accordingly to the SVM.

## Functionalities (not finished)
- Recognizes hand landmarks using Mediapipe solutions
- Gathers information about the hand gesture using the landmarks in your hand
- Using the Support Vector Machine from SK-Learn it generates a prediction model
- Based on this data it can determine if your motion is rock, paper, or scissors

## Libraries used
- Scikit Learn
- Open CV for Python

## Opportunities to strengthen the program
- Use OOP for the repeated CV scripts
- Implement arrays for faster runtime
- Actually do something with the model
