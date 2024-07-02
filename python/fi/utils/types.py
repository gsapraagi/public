from enum import Enum, unique
from typing import List, NamedTuple, Optional, Sequence, TypeVar, Union

import pandas as pd
from fi.utils.logging import get_truncation_warning_message, logger

import numpy as np

from fi.utils.constants import MAX_RAW_DATA_CHARACTERS, MAX_RAW_DATA_CHARACTERS_TRUNCATION


@unique
class ModelTypes(Enum):
    NUMERIC = 1
    SCORE_CATEGORICAL = 2 # Multi class classification, nlp, image classification
    RANKING = 3
    BINARY_CLASSIFICATION = 4 # single class classification
    REGRESSION = 5 # regression, timeseries
    OBJECT_DETECTION = 6
    SEGMENTATION = 7
    GENERATIVE_LLM = 8
    GENERATIVE_IMAGE = 9
    GENERATIVE_VIDEO = 10
    TTS = 11
    STT = 12
    MULTI_MODAL = 13

    @classmethod
    def list_types(cls):
        return [t.name for t in cls]


NUMERIC_MODEL_TYPES = [ModelTypes.NUMERIC, ModelTypes.REGRESSION]
CATEGORICAL_MODEL_TYPES = [
    ModelTypes.SCORE_CATEGORICAL,
    ModelTypes.BINARY_CLASSIFICATION,
]

@unique
class Environments(Enum):
    TRAINING = 1
    VALIDATION = 2
    PRODUCTION = 3
    CORPUS = 4
    


class ObjectDetectionLabel(NamedTuple):
    bounding_boxes_coordinates: List[List[float]]
    categories: List[str]
    scores: Optional[List[float]] = None  # Actual Object Detection Labels won't have scores

    def validate(self, prediction_or_actual: str):
        # Validate bounding boxes
        self._validate_bounding_boxes_coordinates()
        # Validate categories
        self._validate_categories()
        # Validate scores
        self._validate_scores(prediction_or_actual)
        # Validate we have the same number of bounding boxes, categories and scores
        self._validate_count_match()

    def _validate_bounding_boxes_coordinates(self):
        if not is_list_of(self.bounding_boxes_coordinates, list):
            raise TypeError(
                "Object Detection Label bounding boxes must be a list of lists of floats"
            )
        for coordinates in self.bounding_boxes_coordinates:
            if not is_list_of(coordinates, float):
                raise TypeError("Each bounding box's coordinates must be a lists of floats")
            # Format must be (top-left-x, top-left-y, bottom-right-x, bottom-right-y)
            if len(coordinates) != 4:
                raise ValueError(
                    "Each bounding box's coordinates must be a collection of 4 floats. Found "
                    f"{coordinates}"
                )
            if any(coord < 0 for coord in coordinates):
                raise ValueError(
                    f"Bounding box's coordinates cannot be negative. Found {coordinates}"
                )
            if not (coordinates[2] > coordinates[0]):
                raise ValueError(
                    "Each bounding box bottom-right X coordinate should be larger than the "
                    f"top-left. Found {coordinates}"
                )
            if not (coordinates[3] > coordinates[1]):
                raise ValueError(
                    "Each bounding box bottom-right Y coordinate should be larger than the "
                    f"top-left. Found {coordinates}"
                )

    def _validate_categories(self):
        # Allows for categories as empty strings
        if not is_list_of(self.categories, str):
            raise TypeError("Object Detection Label categories must be a list of strings")

    def _validate_scores(self, prediction_or_actual: str):
        if self.scores is None:
            if prediction_or_actual == "prediction":
                raise ValueError("Bounding box confidence scores must not be None for predictions")
        else:
            if prediction_or_actual == "actual":
                raise ValueError("Bounding box confidence scores must be None for actuals")

            if not is_list_of(self.scores, float):
                raise TypeError("Object Detection Label scores must be a list of floats")
            if any(score > 1 or score < 0 for score in self.scores):
                raise ValueError(
                    f"Bounding box confidence scores must be between 0 and 1, inclusive. Found "
                    f"{self.scores}"
                )

    def _validate_count_match(self):
        n_bounding_boxes = len(self.bounding_boxes_coordinates)
        if n_bounding_boxes == 0:
            raise ValueError(
                f"Object Detection Labels must contain at least 1 bounding box. Found"
                f" {n_bounding_boxes}."
            )

        n_categories = len(self.categories)
        if n_bounding_boxes != n_categories:
            raise ValueError(
                "Object Detection Labels must contain the same number of bounding boxes and "
                f"categories. Found {n_bounding_boxes} bounding boxes and {n_categories} "
                "categories."
            )

        if self.scores is not None:
            n_scores = len(self.scores)
            if n_bounding_boxes != n_scores:
                raise ValueError(
                    "Object Detection Labels must contain the same number of bounding boxes and "
                    f"confidence scores. Found {n_bounding_boxes} bounding boxes and {n_scores} "
                    "scores."
                )
            

class RankingPredictionLabel(NamedTuple):
    group_id: str
    rank: int
    score: Optional[float] = None
    label: Optional[str] = None

    def validate(self):
        # Validate existence of required fields: prediction_group_id and rank
        if self.group_id is None or self.rank is None:
            raise ValueError("RankingPredictionLabel must contain: group_id and rank")
        # Validate prediction_group_id
        self._validate_group_id()
        # Validate rank
        self._validate_rank()
        # Validate label type
        if self.label is not None:
            self._validate_label()
        # Validate score type
        if self.score is not None:
            self._validate_score()

    def _validate_group_id(self):
        if not isinstance(self.group_id, str):
            raise TypeError("Prediction Group ID must be a string")
        if not (1 <= len(self.group_id) <= 36):
            raise ValueError(
                f"Prediction Group ID must have length between 1 and 36. Found {len(self.group_id)}"
            )

    def _validate_rank(self):
        if not isinstance(self.rank, int):
            raise TypeError("Prediction Rank must be an int")
        if not (1 <= self.rank <= 100):
            raise ValueError(
                f"Prediction Rank must be between 1 and 100, inclusive. Found {self.rank}"
            )

    def _validate_label(self):
        if not isinstance(self.label, str):
            raise TypeError("Prediction Label must be a str")
        if self.label == "":
            raise ValueError("Prediction Label must not be an empty string.")

    def _validate_score(self):
        if not isinstance(self.score, (float, int)):
            raise TypeError("Prediction Score must be a float or an int")



class RankingActualLabel(NamedTuple):
    relevance_labels: Optional[List[str]] = None
    relevance_score: Optional[float] = None

    def validate(self):
        # Validate relevance_labels type
        if self.relevance_labels is not None:
            self._validate_relevance_labels(self.relevance_labels)
        # Validate relevance score type
        if self.relevance_score is not None:
            self._validate_relevance_score(self.relevance_score)

    @staticmethod
    def _validate_relevance_labels(relevance_labels: List[str]):
        if not is_list_of(relevance_labels, str):
            raise TypeError("Actual Relevance Labels must be a list of strings")
        if any(label == "" for label in relevance_labels):
            raise ValueError("Actual Relevance Labels must be not contain empty strings")

    @staticmethod
    def _validate_relevance_score(relevance_score: float):
        if not isinstance(relevance_score, (float, int)):
            raise TypeError("Actual Relevance score must be a float or an int")
        
        
class Embedding(NamedTuple):
    vector: List[float]
    data: Optional[Union[str, List[str]]] = None
    link_to_data: Optional[str] = None

    def validate(self, emb_name: Union[str, int, float]) -> None:
        """
        Validates that the embedding object passed is of the correct format. That is, validations must
        be passed for vector, data & link_to_data.

        Arguments:
        ----------
            emb_name (str, int, float): Name of the embedding feature the vector belongs to

        Raises:
        -------
            TypeError: If the embedding fields are of the wrong type
        """

        if self.vector is not None:
            self._validate_embedding_vector(emb_name)

        # Validate embedding raw data, if present
        if self.data is not None:
            self._validate_embedding_data(emb_name, self.data)

        # Validate embedding link to data, if present
        if self.link_to_data is not None:
            self._validate_embedding_link_to_data(emb_name, self.link_to_data)

        return None

    def _validate_embedding_vector(
        self,
        emb_name: Union[str, int, float],
    ) -> None:
        """
        Validates that the embedding vector passed is of the correct format. That is:
            1. Type must be list or convertible to list (like numpy arrays, pandas Series)
            2. List must not be empty
            3. Elements in list must be floats

        Arguments:
        ----------
            emb_name (str, int, float): Name of the embedding feature the vector belongs to

        Raises:
        -------
            TypeError: If the embedding does not satisfy requirements above
        """

        if not Embedding._is_valid_iterable(self.vector):
            raise TypeError(
                f'Embedding feature "{emb_name}" has vector type {type(self.vector)}. Must be '
                f"list, "
                f"np.ndarray or pd.Series"
            )
        # Fail if not all elements in list are floats
        allowed_types = (int, float, np.int16, np.int32, np.float16, np.float32)
        if not all(isinstance(val, allowed_types) for val in self.vector):  # type: ignore
            raise TypeError(
                f"Embedding vector must be a vector of integers and/or floats. Got "
                f"{emb_name}.vector = {self.vector}"
            )
        # Fail if the length of the vector is 1
        if len(self.vector) == 1:
            raise ValueError("Embedding vector must not have a size of 1")

    @staticmethod
    def _validate_embedding_data(
        emb_name: Union[str, int, float], data: Union[str, List[str]]
    ) -> None:
        """
        Validates that the embedding raw data field is of the correct format. That is:
            1. Must be string or list of strings (NLP case)

        Arguments:
        ----------
            emb_name (str, int, float): Name of the embedding feature the vector belongs to
            data (str, int, float): Raw data associated with the embedding feature. Typically raw text.

        Raises:
        -------
            TypeError: If the embedding does not satisfy requirements above
        """
        # Validate that data is a string or iterable of strings
        is_string = isinstance(data, str)
        is_allowed_iterable = not is_string and Embedding._is_valid_iterable(data)
        if not (is_string or is_allowed_iterable):
            raise TypeError(
                f'Embedding feature "{emb_name}" data field must be str, list, np.ndarray or '
                f"pd.Series"
            )

        if is_allowed_iterable:
            # Fail if not all elements in iterable are strings
            if not all(isinstance(val, str) for val in data):
                raise TypeError("Embedding data field must contain strings")

        character_count = count_characters_raw_data(data)
        if character_count > MAX_RAW_DATA_CHARACTERS:
            raise ValueError(
                f"Embedding data field must not contain more than {MAX_RAW_DATA_CHARACTERS} characters. "
                f"Found {character_count}."
            )
        elif character_count > MAX_RAW_DATA_CHARACTERS_TRUNCATION:
            logger.warning(
                get_truncation_warning_message(
                    "Embedding raw data fields", MAX_RAW_DATA_CHARACTERS_TRUNCATION
                )
            )

    @staticmethod
    def _validate_embedding_link_to_data(
        emb_name: Union[str, int, float], link_to_data: str
    ) -> None:
        """
        Validates that the embedding link to data field is of the correct format. That is:
            1. Must be string

        Arguments:
        ----------
            emb_name (str, int, float): Name of the embedding feature the vector belongs to
            link_to_data (str): Link to source data of embedding feature, typically an image file on
                cloud storage

        Raises:
        -------
            TypeError: If the embedding does not satisfy requirements above
        """
        if not isinstance(link_to_data, str):
            raise TypeError(
                f'Embedding feature "{emb_name}" link_to_data field must be str and got '
                f"{type(link_to_data)}"
            )

    @staticmethod
    def _is_valid_iterable(data: Union[str, List[str], List[float], np.ndarray, pd.Series]) -> bool:
        """
        Validates that the input data field is of the correct iterable type. That is:
            1. List or
            2. numpy array or
            3. pandas Series

        Arguments:
        ----------
            data: input iterable

        Returns:
        --------
            True if the data type is one of the accepted iterable types, false otherwise
        """
        return any(isinstance(data, t) for t in (list, np.ndarray, pd.Series))


T = TypeVar("T", bound=type)

def is_list_of(lst: Sequence[object], tp: T) -> bool:
    return isinstance(lst, list) and all(isinstance(x, tp) for x in lst)

def count_characters_raw_data(data: Union[str, List[str]]) -> int:
    character_count = 0
    if isinstance(data, str):
        return len(data)
    for string in data:
        character_count += len(string)
    return character_count