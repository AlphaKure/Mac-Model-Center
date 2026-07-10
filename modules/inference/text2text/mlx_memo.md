
Now version 0.31.3 mlx-lm with transformers 5.13^ have bug. 

```
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/Users/alpha/Documents/Mac-Model-Center/modules/inference/text2text/mlx.py", line 1, in <module>
    from mlx_lm import load, generate
  File "/Users/alpha/Documents/Mac-Model-Center/.venv/lib/python3.12/site-packages/mlx_lm/__init__.py", line 9, in <module>
    from .convert import convert
  File "/Users/alpha/Documents/Mac-Model-Center/.venv/lib/python3.12/site-packages/mlx_lm/convert.py", line 11, in <module>
    from .utils import (
  File "/Users/alpha/Documents/Mac-Model-Center/.venv/lib/python3.12/site-packages/mlx_lm/utils.py", line 41, in <module>
    from .tokenizer_utils import TokenizerWrapper
  File "/Users/alpha/Documents/Mac-Model-Center/.venv/lib/python3.12/site-packages/mlx_lm/tokenizer_utils.py", line 505, in <module>
    AutoTokenizer.register("NewlineTokenizer", fast_tokenizer_class=NewlineTokenizer)
  File "/Users/alpha/Documents/Mac-Model-Center/.venv/lib/python3.12/site-packages/transformers/models/auto/tokenization_auto.py", line 984, in register
    TOKENIZER_MAPPING.register(config_class, tokenizer_class, exist_ok=exist_ok)
  File "/Users/alpha/Documents/Mac-Model-Center/.venv/lib/python3.12/site-packages/transformers/models/auto/auto_factory.py", line 680, in register
    if key.__module__.startswith("transformers."):
       ^^^^^^^^^^^^^^
AttributeError: 'str' object has no attribute '__module__'. Did you mean: '__mod__'?
```

Fixed with: https://github.com/ml-explore/mlx-lm/pull/1465