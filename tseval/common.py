from typing import Type, Dict, Any, Union, TypeVar, get_type_hints, get_args, get_origin
from dataclasses import dataclass, is_dataclass

@dataclass
class LLMConfig:
    model_id: str
    model_name: str
    input_cost_per_token: float
    output_cost_per_token: float

@dataclass
class LLMCallRecord:
    model: str
    prompt: str
    response: str
    func_name: str
    input_args: dict
    result: dict
    total_cost: float
    total_time: float
    retry_count: int


@dataclass
class KeywordVerificationResultItem:
    blank: int
    content: str


@dataclass
class KeywordVerificationResult:
    answer: list[KeywordVerificationResultItem]


@dataclass
class StepVerificationResult:
    answer: list[str]


T = TypeVar("T")
def from_dict(cls: Type[T], data: Union[Dict[str, Any], Any]) -> T:
    if isinstance(data, dict):
        if not is_dataclass(cls):
            raise ValueError(f"{cls} is not a dataclass")

        fieldtypes = get_type_hints(cls)
        return cls(**{
            f: from_dict(fieldtypes[f], data[f]) if isinstance(data[f], dict) else (
                [from_dict(get_args(fieldtypes[f])[0], item) for item in data[f]] if get_origin(fieldtypes[f]) is list else data[f]
            )
            for f in data
        })
    else:
        return data
