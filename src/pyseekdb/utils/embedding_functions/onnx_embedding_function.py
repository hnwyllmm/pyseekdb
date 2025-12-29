from pyseekdb.client.embedding_function import EmbeddingFunction, Embeddings, Documents
from typing import Dict, Any, Optional, List
from pathlib import Path
import os
import logging
import shutil
import numpy as np
import numpy.typing as npt
from functools import cached_property
import httpx
import json
import subprocess
import sys

logger = logging.getLogger(__name__)

class OnnxEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    Embedding function using ONNX runtime for sentence-transformers models.
    
    Supports both pre-converted ONNX models and automatic conversion of models
    that don't have ONNX files available. Automatically detects pooling strategy
    and max sequence length from model configuration.

    Example:

        .. code-block:: python
        # Use a model with pre-converted ONNX files
        ef = OnnxEmbeddingFunction("all-MiniLM-L6-v2")
        embeddings = ef(["Hello world", "How are you?"])
        print(len(embeddings[0]))  # 384
        
        # Use any sentence-transformers model (auto-converts if needed)
        ef2 = OnnxEmbeddingFunction("all-mpnet-base-v2", auto_convert=True)
        embeddings2 = ef2(["Hello world", "How are you?"])
        
        # Manually specify pooling strategy and max sequence length
        ef3 = OnnxEmbeddingFunction(
            "all-MiniLM-L6-v2",
            pooling_strategy="mean",
            max_seq_length=256
        )

    """

    _DOWNLOAD_PATH_PREFIX = Path.home() / ".cache" / "pyseekdb" / "onnx_models"
    _DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"

    _MODEL_FILES_TO_DOWNLOAD = {
        "onnx/model.onnx": "model.onnx",  # ONNX file in onnx subdirectory
        "tokenizer.json": "tokenizer.json",
        "config.json": "config.json",
        "special_tokens_map.json": "special_tokens_map.json",
        "tokenizer_config.json": "tokenizer_config.json",
        "vocab.txt": "vocab.txt",
    }

    # Optional files that may exist
    _OPTIONAL_FILES = {
        "modules.json": "modules.json",  # Sentence-transformers specific
        "1_Pooling/config.json": "pooling_config.json",  # Pooling layer config
    }

    def __init__(
        self,
        model_name: str,
        preferred_providers: Optional[List[str]] = None,
        max_seq_length: Optional[int] = None,
        pooling_strategy: Optional[str] = None,
        auto_convert: bool = True,
    ):
        """
        Initialize the ONNX embedding function.

        Args:
            model_name: Name of the model (e.g., 'all-MiniLM-L6-v2', 'all-mpnet-base-v2').
                       Can be a sentence-transformers model name or Hugging Face model ID.
            preferred_providers: The preferred ONNX runtime providers.
                                Defaults to None (uses available providers).
            max_seq_length: Maximum sequence length. If None, will be auto-detected from model config.
            pooling_strategy: Pooling strategy ('mean', 'cls', 'max', 'pooler'). 
                            If None, will be auto-detected from model config.
            auto_convert: If True, automatically convert models to ONNX if pre-converted 
                         ONNX files are not available. Requires optimum library.
        """
        if not model_name:
            raise ValueError("Model name is required")
        self.model_name = model_name
        # Support both sentence-transformers/ prefix and direct model names
        if model_name.startswith("sentence-transformers/"):
            self.hf_model_id = model_name
            self.model_name_short = model_name.replace("sentence-transformers/", "")
        else:
            self.hf_model_id = f"sentence-transformers/{model_name}"
            self.model_name_short = model_name
        
        self._max_seq_length_param = max_seq_length
        self._pooling_strategy_param = pooling_strategy
        self._auto_convert = auto_convert

        # Validate preferred_providers
        if preferred_providers and not all(
            [isinstance(i, str) for i in preferred_providers]
        ):
            raise ValueError("Preferred providers must be a list of strings")
        if preferred_providers and len(preferred_providers) != len(
            set(preferred_providers)
        ):
            raise ValueError("Preferred providers must be unique")

        self._preferred_providers = preferred_providers

        # Import required modules
        import onnxruntime as ort_module
        import tokenizers
        import tqdm

        self.ort = ort_module
        self.tokenizers = tokenizers  # Store the module
        self.tqdm = tqdm.tqdm

    def _download(self, url: str, fname: Path, chunk_size: int = 8192) -> None:
        """
        Download a file from the URL and save it to the file path.

        Args:
            url: The URL to download the file from.
            fname: The path to save the file to.
            chunk_size: The chunk size to use when downloading (default: 8192 for better speed).
        """
        logger.info(f"Downloading from {url}")
        # Use Client to ensure correct handling of redirects
        with httpx.Client(timeout=600.0, follow_redirects=True) as client:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                with open(fname, "wb") as file, self.tqdm(
                    desc=fname.name,
                    total=total,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in resp.iter_bytes(chunk_size=chunk_size):
                        size = file.write(data)
                        bar.update(size)

    def _get_hf_endpoint(self) -> str:
        """Get Hugging Face endpoint URL, using HF_ENDPOINT environment variable if set."""
        return os.environ.get("HF_ENDPOINT", self._DEFAULT_HF_ENDPOINT)

    def _download_path(self) -> Path:
        """Get the path to the download directory."""
        return self._DOWNLOAD_PATH_PREFIX / self.model_name

    def _download_tmp_path(self) -> Path:
        download_path = self._download_path()
        return download_path.parent / (download_path.name + ".tmp")

    def _download_file_from_hf(self, hf_filename: str, local_path: Path, hf_endpoint: str) -> bool:
        """
        Download a single file from Hugging Face.
        
        Returns:
            True if successful, False if file not found (404).
        """
        url = f"{hf_endpoint}/{self.hf_model_id}/resolve/main/{hf_filename}"
        
        try:
            # First check if file exists (HEAD request)
            try:
                head_resp = httpx.head(url, timeout=10.0, follow_redirects=True)
                if head_resp.status_code == 404:
                    return False
            except Exception:
                # If HEAD request fails, continue with GET request
                pass

            self._download(url, local_path, chunk_size=8192)
            logger.info(f"Successfully downloaded {local_path.name}")
            return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return False
            else:
                raise RuntimeError(f"HTTP error downloading {hf_filename} from Hugging Face: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to download {hf_filename} from Hugging Face: {e}") from e

    def _download_from_huggingface(self) -> bool:
        """
        Download model files from Hugging Face (supports mirror acceleration).

        Returns:
            True if download successful, False if ONNX files not found.
        """
        extracted_folder_tmp = self._download_tmp_path()
        hf_endpoint = self._get_hf_endpoint().rstrip('/')

        try:
            shutil.rmtree(extracted_folder_tmp, ignore_errors=True)
            extracted_folder_tmp.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading model from Hugging Face (endpoint: {hf_endpoint})")

            # Try to download ONNX model first
            onnx_downloaded = self._download_file_from_hf(
                "onnx/model.onnx",
                extracted_folder_tmp / "model.onnx",
                hf_endpoint
            )
            
            if not onnx_downloaded:
                logger.warning("ONNX model not found on Hugging Face, will try conversion")
                shutil.rmtree(extracted_folder_tmp, ignore_errors=True)
                return False

            # Download required files
            required_files_downloaded = True
            for hf_filename, local_filename in self._MODEL_FILES_TO_DOWNLOAD.items():
                if hf_filename == "onnx/model.onnx":
                    continue  # Already downloaded
                    
                local_path = extracted_folder_tmp / local_filename
                if not self._download_file_from_hf(hf_filename, local_path, hf_endpoint):
                    if hf_filename in ["tokenizer.json", "config.json"]:
                        # These are critical
                        logger.error(f"Critical file {hf_filename} not found")
                        shutil.rmtree(extracted_folder_tmp, ignore_errors=True)
                        return False
                    else:
                        logger.warning(f"Optional file {hf_filename} not found, continuing...")

            # Try to download optional files
            for hf_filename, local_filename in self._OPTIONAL_FILES.items():
                local_path = extracted_folder_tmp / local_filename
                self._download_file_from_hf(hf_filename, local_path, hf_endpoint)

            # Check whether the key files exist
            if not (extracted_folder_tmp / "model.onnx").exists():
                shutil.rmtree(extracted_folder_tmp, ignore_errors=True)
                return False
            if not (extracted_folder_tmp / "tokenizer.json").exists():
                shutil.rmtree(extracted_folder_tmp, ignore_errors=True)
                return False

            logger.info("Successfully downloaded all model files from Hugging Face")
            shutil.rmtree(self._download_path(), ignore_errors=True)
            extracted_folder_tmp.rename(self._download_path())
            return True

        except Exception as e:
            shutil.rmtree(extracted_folder_tmp, ignore_errors=True)
            raise RuntimeError(f"Download model failed. Model ID: {self.hf_model_id}\n" \
                            f"endpoint: {hf_endpoint}\n" \
                            f"download path: {extracted_folder_tmp}\n" \
                            f"error: {e}") from e

    def _apply_pooling(
        self,
        last_hidden_state: npt.NDArray[np.float32],
        attention_mask: npt.NDArray[np.int64],
        pooling_strategy: str,
    ) -> npt.NDArray[np.float32]:
        """
        Apply pooling strategy to get sentence embeddings.

        Args:
            last_hidden_state: The last hidden state from the model (batch_size, seq_length, hidden_size)
            attention_mask: The attention mask (batch_size, seq_length)
            pooling_strategy: The pooling strategy ('mean', 'cls', 'max', 'pooler')

        Returns:
            Pooled embeddings (batch_size, hidden_size)
        """
        if pooling_strategy == "mean":
            # Mean pooling (weighted by attention mask)
            attention_mask_float = attention_mask.astype(np.float32)
            input_mask_expanded = np.broadcast_to(
                np.expand_dims(attention_mask_float, -1), last_hidden_state.shape
            )
            embeddings = np.sum(last_hidden_state * input_mask_expanded, 1) / np.clip(
                input_mask_expanded.sum(1), a_min=1e-9, a_max=None
            )
        elif pooling_strategy == "cls":
            # CLS token pooling (first token)
            embeddings = last_hidden_state[:, 0, :]
        elif pooling_strategy == "max":
            # Max pooling
            attention_mask_float = attention_mask.astype(np.float32)
            input_mask_expanded = np.broadcast_to(
                np.expand_dims(attention_mask_float, -1), last_hidden_state.shape
            )
            # Set padding tokens to very negative values so they don't affect max
            masked_hidden = np.where(
                input_mask_expanded > 0, last_hidden_state, np.finfo(np.float32).min
            )
            embeddings = np.max(masked_hidden, axis=1)
        elif pooling_strategy == "pooler":
            # Use pooler output if available (usually first output after last_hidden_state)
            # For now, fall back to CLS token
            embeddings = last_hidden_state[:, 0, :]
        else:
            raise ValueError(
                f"Unknown pooling strategy: {pooling_strategy}. "
                f"Supported strategies: 'mean', 'cls', 'max', 'pooler'"
            )

        return embeddings.astype(np.float32)

    def _forward(
        self, documents: List[str], batch_size: int = 32
    ) -> npt.NDArray[np.float32]:
        """
        Generate embeddings for a list of documents.

        Args:
            documents: The documents to generate embeddings for.
            batch_size: The batch size to use when generating embeddings.

        Returns:
            The embeddings for the documents.
        """
        all_embeddings = []
        pooling_strategy = self._get_pooling_strategy()
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            # Encode each document separately
            encoded = [self.tokenizer.encode(d) for d in batch]

            # Check if any document exceeds the max tokens
            max_tokens = self._max_tokens()
            for doc_tokens in encoded:
                if len(doc_tokens.ids) > max_tokens:
                    raise ValueError(
                        f"Document length {len(doc_tokens.ids)} is greater than "
                        f"the max tokens {max_tokens}"
                    )

            # Create input arrays, ensuring int64 type
            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

            # Ensure 2D arrays (batch_size, seq_length)
            if input_ids.ndim == 1:
                input_ids = input_ids.reshape(1, -1)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.reshape(1, -1)

            # Use zeros_like to create token_type_ids, ensuring exact shape match
            token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

            # Ensure all arrays are contiguous, which is important for onnxruntime 1.19.0
            input_ids = np.ascontiguousarray(input_ids, dtype=np.int64)
            attention_mask = np.ascontiguousarray(attention_mask, dtype=np.int64)
            token_type_ids = np.ascontiguousarray(token_type_ids, dtype=np.int64)

            onnx_input = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
            }

            model_output = self.model.run(None, onnx_input)
            last_hidden_state = model_output[0]

            # Apply pooling strategy
            embeddings = self._apply_pooling(last_hidden_state, attention_mask, pooling_strategy)
            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings)

    def _get_pooling_strategy(self) -> str:
        """Detect pooling strategy from model config or use default."""
        if self._pooling_strategy_param:
            return self._pooling_strategy_param
        
        download_path = self._download_path()
        
        # Try to read from pooling config
        pooling_config_path = download_path / "pooling_config.json"
        if pooling_config_path.exists():
            try:
                with open(pooling_config_path, 'r') as f:
                    pooling_config = json.load(f)
                    pooling_mode = pooling_config.get("pooling_mode", "mean")
                    if pooling_mode == "mean":
                        return "mean"
                    elif pooling_mode == "cls":
                        return "cls"
                    elif pooling_mode == "max":
                        return "max"
            except Exception as e:
                logger.warning(f"Failed to read pooling config: {e}")
        
        # Try to read from modules.json (sentence-transformers specific)
        modules_json_path = download_path / "modules.json"
        if modules_json_path.exists():
            try:
                with open(modules_json_path, 'r') as f:
                    modules = json.load(f)
                    for module in modules:
                        if module.get("type") == "sentence_transformers.models.Pooling":
                            pooling_mode = module.get("pooling_mode_cls_token", False)
                            pooling_mode_mean = module.get("pooling_mode_mean_tokens", True)
                            pooling_mode_max = module.get("pooling_mode_max_tokens", False)
                            
                            if pooling_mode:
                                return "cls"
                            elif pooling_mode_max:
                                return "max"
                            elif pooling_mode_mean:
                                return "mean"
            except Exception as e:
                logger.warning(f"Failed to read modules.json: {e}")
        
        # Default to mean pooling (most common)
        logger.info("Pooling strategy not detected, defaulting to 'mean'")
        return "mean"

    def _get_max_seq_length(self) -> int:
        """Detect max sequence length from model config or use default."""
        if self._max_seq_length_param:
            return self._max_seq_length_param
        
        download_path = self._download_path()
        config_path = download_path / "config.json"
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Try different possible keys
                    max_length = (
                        config.get("max_position_embeddings") or
                        config.get("max_seq_length") or
                        config.get("model_max_length")
                    )
                    if max_length:
                        # Sentence-transformers often uses a value slightly less than max_position_embeddings
                        # Common values: 128, 256, 384, 512
                        if max_length >= 512:
                            return 512
                        elif max_length >= 384:
                            return 384
                        elif max_length >= 256:
                            return 256
                        elif max_length >= 128:
                            return 128
                        else:
                            return max_length
            except Exception as e:
                logger.warning(f"Failed to read config.json: {e}")
        
        # Default to 256 (common for many models)
        logger.info("Max sequence length not detected, defaulting to 256")
        return 256

    @cached_property
    def tokenizer(self) -> Any:
        """
        Get the tokenizer for the model.

        Returns:
            The tokenizer for the model.
        """
        tokenizer = self.tokenizers.Tokenizer.from_file(
            str(self._download_path() / "tokenizer.json")
        )
        max_length = self._get_max_seq_length()
        tokenizer.enable_truncation(max_length=max_length)
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=max_length)
        return tokenizer

    @cached_property
    def model(self) -> Any:
        """
        Get the model.

        Returns:
            The model.
        """
        if self._preferred_providers is None or len(self._preferred_providers) == 0:
            if len(self.ort.get_available_providers()) > 0:
                logger.debug(
                    f"WARNING: No ONNX providers provided, defaulting to available providers: "
                    f"{self.ort.get_available_providers()}"
                )
            self._preferred_providers = self.ort.get_available_providers()
        elif not set(self._preferred_providers).issubset(
            set(self.ort.get_available_providers())
        ):
            raise ValueError(
                f"Preferred providers must be subset of available providers: "
                f"{self.ort.get_available_providers()}"
            )

        # Create minimal session options to avoid issues
        so = self.ort.SessionOptions()
        so.log_severity_level = 3
        # Disable all optimizations that might cause issues
        so.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.execution_mode = self.ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1

        if (
            self._preferred_providers
            and "CoreMLExecutionProvider" in self._preferred_providers
        ):
            # remove CoreMLExecutionProvider from the list, it is not as well optimized as CPU.
            self._preferred_providers.remove("CoreMLExecutionProvider")

        return self.ort.InferenceSession(
            str(self._download_path() / "model.onnx"),
            # Force CPU execution provider to avoid provider issues
            providers=['CPUExecutionProvider'],
            sess_options=so,
        )

    def _download_model_if_not_exists(self) -> None:
        """
        Download from Hugging Face with image mirror if the model doesn't exist.
        If ONNX files are not available and auto_convert is True, convert the model.
        """
        extracted_folder = self._download_path()
        
        # Check if model.onnx and tokenizer.json exist (critical files)
        onnx_exists = (extracted_folder / "model.onnx").exists()
        tokenizer_exists = (extracted_folder / "tokenizer.json").exists()
        
        if onnx_exists and tokenizer_exists:
            logger.debug(f"Model already exists at {extracted_folder}")
            return

        # Try to download pre-converted ONNX model
        logger.info("Attempting to download pre-converted ONNX model from Hugging Face...")
        download_success = self._download_from_huggingface()
        
        if download_success:
            logger.info("Model downloaded successfully from Hugging Face")
            return
        
        # If download failed and auto_convert is enabled, try conversion
        if self._auto_convert:
            logger.info("Pre-converted ONNX model not found, attempting automatic conversion...")
            self._convert_to_onnx()
            logger.info("Model converted successfully to ONNX")
        else:
            raise RuntimeError(
                f"ONNX model not found for {self.hf_model_id} on Hugging Face. "
                f"Set auto_convert=True to automatically convert the model, or ensure "
                f"the model has pre-converted ONNX files available."
            )

    def _max_tokens(self) -> int:
        """Get the maximum number of tokens supported by the model."""
        return self._get_max_seq_length()

    def _convert_to_onnx(self) -> None:
        """
        Convert a sentence-transformers model to ONNX format using optimum.
        
        This method downloads the model, converts it to ONNX, and saves it locally.
        """
        if not self._auto_convert:
            raise RuntimeError(
                f"ONNX model not found for {self.hf_model_id} and auto_convert is False. "
                f"Please set auto_convert=True or provide a model with pre-converted ONNX files."
            )
        
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "optimum library is required for automatic ONNX conversion. "
                "Install it with: pip install optimum[onnx]"
            )
        
        download_path = self._download_path()
        download_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Converting model {self.hf_model_id} to ONNX format...")
        
        try:
            # Download and convert model
            model = ORTModelForFeatureExtraction.from_pretrained(
                self.hf_model_id,
                export=True,
                provider="CPUExecutionProvider",
            )
            
            # Save ONNX model
            model.save_pretrained(str(download_path))
            
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
            tokenizer.save_pretrained(str(download_path))
            
            # Move model.onnx to the expected location
            onnx_files = list(download_path.glob("*.onnx"))
            if onnx_files:
                # Usually the main model file is the largest or has a specific name
                main_onnx = None
                for onnx_file in onnx_files:
                    if "model" in onnx_file.stem.lower() and "decoder" not in onnx_file.stem.lower():
                        main_onnx = onnx_file
                        break
                
                if main_onnx is None:
                    # Use the largest file
                    main_onnx = max(onnx_files, key=lambda f: f.stat().st_size)
                
                target_path = download_path / "model.onnx"
                if main_onnx != target_path:
                    shutil.move(str(main_onnx), str(target_path))
                    logger.info(f"Moved {main_onnx.name} to model.onnx")
            
            # Ensure tokenizer.json exists (optimum should have saved it)
            if not (download_path / "tokenizer.json").exists():
                # Try to find it in subdirectories or convert from other formats
                tokenizer_files = list(download_path.glob("**/tokenizer.json"))
                if tokenizer_files:
                    shutil.move(str(tokenizer_files[0]), str(download_path / "tokenizer.json"))
            
            logger.info(f"Successfully converted and saved model to {download_path}")
            
        except Exception as e:
            shutil.rmtree(download_path, ignore_errors=True)
            raise RuntimeError(
                f"Failed to convert model {self.hf_model_id} to ONNX: {e}\n"
                f"Make sure the model is compatible with ONNX conversion."
            ) from e

    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for the given documents.

        Args:
            input: Single document (str) or list of documents (List[str])

        Returns:
            List of embedding vectors
        """
        # Handle single string input
        if isinstance(input, str):
            input = [input]

        # Handle empty input
        if not input:
            return []

        # Only download the model when it is actually used
        self._download_model_if_not_exists()

        # Generate embeddings
        embeddings = self._forward(input)

        # Convert numpy arrays to lists
        return [embedding.tolist() for embedding in embeddings]

    def __repr__(self) -> str:
        return f"OnnxEmbeddingFunction(model_name='{self.model_name}', auto_convert={self._auto_convert})"