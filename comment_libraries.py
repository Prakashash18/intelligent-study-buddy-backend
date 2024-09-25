# List of libraries to comment out
libs_to_comment = [
    "regex", "python-magic", "nltk", "google-cloud-vision", "kubernetes", 
    "typing-inspect", "requests-oauthlib", "effdet", "distro", 
    "python-docx", "annotated-types", "pandas", "et-xmlfile", "filelock", 
    "pydantic-core", "jsonpointer", "pdfminer-six", "nest-asyncio", 
    "pyparsing", "duckduckgo-search", "safetensors", "python-iso639", 
    "starlette", "omegaconf", "googleapis-common-protos", 
    "opentelemetry-exporter-otlp-proto-grpc", "langsmith", "kiwisolver", 
    "onnxruntime", "portalocker", "filetype", "uvloop", "markdown-it-py", 
    "oauthlib", "requests-toolbelt", "requests", "watchfiles", 
    "google-resumable-media", "opentelemetry-util-http", 
    "pyreqwest-impersonate", "tenacity", "google-cloud-firestore", "six", 
    "deepdiff", "opentelemetry-exporter-otlp-proto-common", "mmh3", 
    "langchain-text-splitters", "networkx", "onnx", "pyasn1", "emoji", 
    "google-generativeai", "opentelemetry-proto", "uritemplate", 
    "jsonpatch", "opentelemetry-api", "urllib3", "jsonpath-python", 
    "mpmath", "python-dateutil", "pillow-heif", "rapidfuzz", "tabulate", 
    "click", "coloredlogs", "joblib", "python-pptx", "posthog", 
    "cachecontrol", "opentelemetry-instrumentation-fastapi", "pyasn1-modules", 
    "tiktoken", "multidict", "rsa", "aiosignal", "importlib-metadata", 
    "layoutparser", "websocket-client", "fsspec", "certifi", 
    "opentelemetry-sdk", "pyjwt", "charset-normalizer", "pygments", 
    "pypandoc", "rich", "pyyaml", "google-auth-httplib2", "jinja2", 
    "pytesseract", "wheel", "tzdata", "ordered-set", "httplib2", 
    "typing-extensions", "beautifulsoup4", "pycocotools", "typer", 
    "shellingham", "fastapi-cli", "deprecated", "unstructured-client", 
    "packaging", "unstructured-inference", "setuptools", "xlsxwriter", 
    "monotonic", "google-auth", "unstructured-pytesseract", "pikepdf", 
    "timm", "iopath", "msgpack", "pyproject-hooks", "overrides", "openai", 
    "cffi", "langdetect", "opentelemetry-semantic-conventions", "uvicorn", 
    "orjson", "email-validator", "dnspython", "chroma-hnswlib", 
    "marshmallow", "pypika", "scipy", "mdurl", "numpy", "pillow", 
    "markupsafe", "google-api-python-client", "pdfplumber", "pypdf", 
    "yarl", "matplotlib", "chromadb", "chardet", "sniffio", "google-cloud-core", 
    "proto-plus", "contourpy", "tqdm", "google-cloud-storage", "lxml", 
    "bcrypt", "opencv-python", "google-crc32c", "fonttools", "openpyxl", 
    "backoff", "python-oxmsg", "python-multipart", "tokenizers", 
    "antlr4-python3-runtime", "huggingface-hub", "mypy-extensions", 
    "xlrd", "markdown", "docx2txt", "anyio", "cachetools", 
    "google-ai-generativelanguage", "idna", "opentelemetry-instrumentation-asgi", 
    "grpcio-status", "sqlalchemy", "youtube-search", "httptools", "wrapt", 
    "pdf2image", "httpx", "humanfriendly", "transformers", "build", 
    "ujson", "attrs", "h11", "frozenlist", "sympy", "httpcore", 
    "dataclasses-json", "torchvision", "opentelemetry-instrumentation", 
    "websockets", "pycparser", "zipp", "flatbuffers", "importlib-resources", 
    "protobuf", "google-api-core", "asgiref", "cryptography", "aiohttp", 
    "pypdfium2", "torch", "soupsieve", "unstructured", "cycler", "pytz", 
    "grpcio", "olefile"
]

# Read the current requirements.txt
with open('requirements.txt', 'r') as file:
    lines = file.readlines()

# Comment out the required libraries
with open('requirements.txt', 'w') as file:
    for line in lines:
        package_name = line.split('==')[0].strip()
        if package_name in libs_to_comment:
            file.write(f"# {line}")
        else:
            file.write(line)

print("Libraries have been commented out successfully.")
