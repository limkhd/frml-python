import logging
import sys

import data_prep
import frml_python.model
import frml_python.evaluator

format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=format)

root_logger = logging.getLogger()

# create file handler which logs even debug messages
fh = logging.FileHandler("main.log")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter(format))
root_logger.addHandler(fh)

logger = logging.getLogger(__name__)


def main():
    data_path = "../data/IN300_folds.mat"
    logger.info("Loading data from %s" % data_path)
    folds = data_prep.load_folds(data_path)

    # Use Fold #0
    fold_index = 0

    # max number of similar examples in training set
    train_k = 5

    mlo = data_prep.get_ranking_labeled_ML_obj(folds, fold_index, train_k)

    logger.info("Initializing model")
    frm = frml_python.model.FRML_model(
        num_iters=100000,
        stepsize=10000,
        d=2,
        lam=0.7,
        batchsize=2,
        report_interval=2500,
        Xval=mlo["Xval"],
        Yval=mlo["Yval"],
        WARP_sampling_limit=5,
        random_state=0,
        warm_start_L=None,
    )

    logger.info("Fitting model")
    frm.fit(mlo["Xtrain"], mlo["Ytrain"])

    evaluator_ = frml_python.evaluator.Evaluator(mlo["Xtrain"], mlo["Ytrain"])
    logger.info("Evaluating model on test set")
    test_score = evaluator_.evaluate_ranking(
        frm, mlo["Xtrain"], mlo["Ytrain"], mlo["Xtest"], mlo["Ytest"]
    )

    logger.info("Test MAP is %.3f" % test_score)


if __name__ == "__main__":
    main()
