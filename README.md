# Mac Model Center

## Install 

1. Clone project
```
git clone https://github.com/AlphaKure/Mac-Model-Center.git
```

2. Create enviroment
```
uv venv -p python3.12
```

3. Install require modules
```
uv pip sync uv.lock
```

4. Run Code
```
python main.py
# or
uvicorn main:app --host <HOST> --port <port>
```

## Feature and support models

Only list tested models.

- [Text to Text]
  
     - [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) 

- [Text to Image]

    - [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)

    - [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo)
