import os.path

import causal_model_generator
import question0
import question1
import question2
import question3
import questions_utils


def __main__():
    if not os.path.exists("../mubench/data/mubench_df.csv"):
        causal_model_generator.read_experiments("../mubench/data/", {s: s for s in questions_utils.services})
    question0.__main__()
    question1.__main__()
    question2.__main__()
    question3.__main__()


__main__()
