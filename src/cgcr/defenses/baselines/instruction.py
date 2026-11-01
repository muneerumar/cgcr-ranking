"""reimplementation: instruction defense; append a fixed ranking instruction.

Generic instruction baseline from the project roadmap, not an official model.
The instruction is versioned prompt data and contains no attack material.
"""
from pathlib import Path


def instruction_template(ranking_template, instruction_path):
    return ranking_template + '\n' + Path(instruction_path).read_text()
