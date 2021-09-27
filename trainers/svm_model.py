from sklearn import svm
from get_data import capture_data
import pickle

"""
Run this script to start capturing your own data for rock, paper, scissors.
- Pose your hand as the on-screen text prompts to, be aware of when it asks for another gesture
- The script pickles your Support Vector Machine model with name 'rsp_model.sav'
"""

def main():
    X1, y1 = capture_data("Rock", 1)
    X2, y2 = capture_data("Palm", 0)
    X3, y3 = capture_data("Scissors", 2)

    X, y = X1 + X2 + X3, y1 + y2 + y3

    clf = svm.SVC()
    clf.fit(X, y)
    pickle.dump(clf, open("rsp_model.sav", 'wb'))

if __name__ == "__main__":
    main()