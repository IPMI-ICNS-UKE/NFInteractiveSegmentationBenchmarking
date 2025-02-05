# This file contains code from the SW-FastEdit repository:
# https://github.com/Zrrr1997/SW-FastEdit
#
# The original implementation accompanies the following research paper:
#
# M. Hadlich, Z. Marinov, M. Kim, E. Nasca, J. Kleesiek, and R. Stiefelhagen,
# "Sliding Window Fastedit: A Framework for Lesion Annotation in Whole-Body PET Images,"
# 2024 IEEE International Symposium on Biomedical Imaging (ISBI), Athens, Greece, 2024, 
# pp. 1-5, doi: 10.1109/ISBI56570.2024.10635459.
#
# Keywords: Training; Image segmentation; Solid modeling; Annotations; Memory management;
# Whole-body PET; Manuals; Interactive Segmentation; PET; Sliding Window; 
# Lung Cancer; Melanoma; Lymphoma

from __future__ import annotations

import logging

logger = None


def setup_loggers(loglevel=logging.INFO, log_file_folder=None):
    global logger
    logger = logging.getLogger("evaluation_pipeline_logger")
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # Add the stream handler
    stream_handler = logging.StreamHandler()
    # (%(name)s)
    formatter = logging.Formatter(
        fmt="[%(asctime)s.%(msecs)03d][%(levelname)s] %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(loglevel)
    logger.addHandler(stream_handler)
    file_handler = None

    if log_file_folder is not None:
        # Add the file handler
        log_file_path = f"{log_file_folder}/log.txt"
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(loglevel)
        logger.addHandler(file_handler)
        logger.info(f"Logging all the data to '{log_file_path}'")
    else:
        logger.info("Logging only to the console")

    # Set logging level for external libraries
    for _ in (
        "ignite.engine.engine.SupervisedTrainer",
        "ignite.engine.engine.SupervisedEvaluator",
    ):
        l = logging.getLogger(_)
        if l.hasHandlers():
            l.handlers.clear()
        l.propagate = False
        l.setLevel(loglevel)
        l.addHandler(stream_handler)
        if file_handler is not None:
            l.addHandler(file_handler)


def get_logger():
    global logger
    if logger is None:
        raise UserWarning("Logger not initialized")
    else:
        return logger
