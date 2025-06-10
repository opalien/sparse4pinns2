from typing import Any, cast
import json
import os
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

# save to format js line
def save_result(path: str, results: dict[str, Any]):
    results_str = json.dumps(results, cls=NumpyEncoder)
    
    # create directory if it does not exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'a') as f:
        f.write(results_str + '\n')
    


def load_results(path: str) -> list[dict[str, Any]]:
    list_results: list[dict[str, Any]] = []
    with open(path, 'r') as f:
        for line in f:
            ljs = json.loads(line)

            if not isinstance(ljs, dict):
                raise ValueError(f"Invalid line in file: {line}")
            
            typed_ljs = cast(dict[str, Any], ljs)
            list_results.append(typed_ljs)
    return list_results


def save_results(path: str, results: list[dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'a') as f:
        for result in results:
            result_str = json.dumps(result)
            f.write(result_str + '\n')


def conc_files(input_file: str, output_file: str):
    with open(input_file, 'r') as f_in, open(output_file, 'a') as f_out:
        for line in f_in:
            f_out.write(line)