from sklearn import svm
import numpy as np
import pandas as pd


def load_ir(pages_relevance_file, paragraph_relevance_file, dot2sum_file="DotToSum-Training-Weighted.csv",
        sum2dot_file="SumToDot-Training-Weighted.csv"):
    file_it = iter(zip(open(pages_relevance_file), open(paragraph_relevance_file)))
    data = list()
    next(file_it)
    for page_line, para_line in file_it:
        page_line = page_line.strip()
        page_line = page_line.split(",")
        para_line = para_line.strip()
        para_line = para_line.split(",")
        feature_vector = [float(e) for e in page_line[1:]] + [float(e) for e in para_line[1:]]
        data.append(feature_vector)
    DotToSum = pd.read_csv(dot2sum_file)
    SumToDot = pd.read_csv(sum2dot_file)

    print(np.array(data).shape, SumToDot.shape)

    data = np.concatenate((np.array(data), SumToDot), axis=1)

    data =  np.concatenate((np.array(data), DotToSum), axis=1)
    return data

def load_target(training_file="training_set.tsv"):
    target = list()
    possible_answers = ["A", "B", "C", "D"]
    fobj = open(training_file)
    next(fobj)
    for line in fobj:
        line = line.strip()
        line = line.split("\t")
        answer = line[2]
        target.append(possible_answers.index(answer))
    return target

def contains_underscores(training_file, data):
    """
    1 if _____ is in question
    this feature is not used
    function modifies data input to add this column
    """
    contains_underscores = list()
    fobj = open(training_file)
    next(fobj)
    for line in fobj:
        line = line.strip()
        line = line.split("\t")
        contains_underscores.append(int("____" in line[1]))
    contains_underscores = np.array(contains_underscores)
    data = np.concatenate((np.array(data), np.array(list([contains_underscores])).T), axis=1)
    return data

def cross_validation(data, target):
    """
    to run cross validation, data is training data of course
    """
    from sklearn import cross_validation
    clf = svm.SVC(kernel='linear')
    scores = cross_validation.cross_val_score(clf, data, target, cv=5, n_jobs=-1)
    print(scores)
    print("cross validation score = %s" % np.mean(scores))

if __name__ == "__main__":
    train_pages_relevance_file = "tsdf.tmp.csv"
    train_paragraph_relevance_file = "paragraphsdf.tmp.csv"
    training_file = "training_set.tsv"
    train_dot2sum_file="DotToSum-Training-Weighted.csv"
    train_sum2dot_file="SumToDot-Training-Weighted.csv"

    test_pages_relevance_file = "pagetestsdf.tmp.csv"
    test_paragraph_relevance_file = "paratestsdf.tmp.csv"
    test_dot2sum_file = "DotToSum-Test-Weighted.csv"
    test_sum2dot_file = "SumToDot-Test-Weighted.csv"

    prediction_file = "output.csv"

    train_data = load_ir(train_pages_relevance_file,
                         train_paragraph_relevance_file,
                         dot2sum_file=train_dot2sum_file,
                         sum2dot_file=train_sum2dot_file)


    target = load_target(training_file=training_file)
    print("training data was loaded")

    test_data = load_ir(test_pages_relevance_file,
                        test_paragraph_relevance_file,
                        dot2sum_file=test_dot2sum_file,
                        sum2dot_file=test_sum2dot_file)

    print("test data was loaded")

    clf = svm.SVC(kernel='linear', verbose=False, max_iter=10e5)
    print("start training")
    clf.fit(train_data, target)
    print("Training, done. Start prediction")

    predictions = clf.predict(test_data)
    i = 102501
    with open(prediction_file, "w") as f:
        f.write("id,correctAnswer\n")
        for j, pred in enumerate(predictions):
            f.write("%s,%s\n" % (i+j,["A","B","C","D"][pred]))
