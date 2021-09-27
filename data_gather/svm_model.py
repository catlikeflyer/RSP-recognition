from sklearn import svm
from get_data import capture_data

def main():
    X, y = capture_data()

    clf = svm.SVC()
    clf.fit(X, y)

if __name__ == "__main__":
    main()