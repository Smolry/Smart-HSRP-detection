import os
import torch
import cv2
import numpy as np
from pathlib import Path
from config.settings import settings


class HSRPClassifier:
    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, model_path: str = settings.HSRP_MODEL_PATH):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = float(settings.HSRP_CONF_THRESHOLD)
        self.using_trt = False
        self.trt_ctx   = None
        self.model     = None

        engine_path = str(Path(model_path).with_suffix(".engine"))

        if self.device.type == "cuda" and os.path.exists(engine_path):
            try:
                self._load_trt_engine(engine_path)
                self.using_trt = True
                print("[HSRPClassifier] TensorRT engine loaded | FP16")
            except Exception as e:
                print(f"[HSRPClassifier] TRT load failed ({e}), falling back to TorchScript")
                self._load_torchscript(model_path)
        else:
            self._load_torchscript(model_path)

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self._mean = self._MEAN.to(self.device, dtype=dtype)
        self._std  = self._STD.to(self.device, dtype=dtype)
        print(f"[HSRPClassifier] {self.device} | TRT={self.using_trt}")

    def _load_trt_engine(self, engine_path: str):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.trt_engine = runtime.deserialize_cuda_engine(f.read())
        self.trt_ctx = self.trt_engine.create_execution_context()
        self._cuda = cuda
        self._input_shape  = (1, 3, 224, 224)
        self._output_shape = (1, 1)
        self._d_input  = cuda.mem_alloc(int(np.prod(self._input_shape))  * 2)
        self._d_output = cuda.mem_alloc(int(np.prod(self._output_shape)) * 2)
        self._stream   = cuda.Stream()

    def _load_torchscript(self, model_path: str):
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            if self.device.type == "cuda":
                self.model = self.model.half()
            print("[HSRPClassifier] TorchScript loaded")
        except Exception as e:
            print(f"[HSRPClassifier] TorchScript load failed: {e}")
            self.model = None

    def preprocess(self, img: np.ndarray) -> torch.Tensor:
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t   = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t   = t.to(self.device)
        if self.device.type == "cuda":
            t = t.half()
        return (t - self._mean) / self._std

    def _trt_infer(self, x: torch.Tensor) -> float:
        arr = x.contiguous().cpu().numpy().astype(np.float16)
        self._cuda.memcpy_htod_async(self._d_input, arr, self._stream)
        # TensorRT 10.x API: set tensor addresses then execute
        self.trt_ctx.set_tensor_address("input",  int(self._d_input))
        self.trt_ctx.set_tensor_address("output", int(self._d_output))
        self.trt_ctx.execute_async_v3(stream_handle=self._stream.handle)
        out = np.empty(self._output_shape, dtype=np.float16)
        self._cuda.memcpy_dtoh_async(out, self._d_output, self._stream)
        self._stream.synchronize()
        return float(1.0 / (1.0 + np.exp(-float(out[0, 0]))))

    def predict(self, plate_image: np.ndarray) -> dict:
        if plate_image is None or plate_image.size == 0:
            return self._empty()
        x = self.preprocess(plate_image)
        if self.using_trt and self.trt_ctx:
            prob_non_hsrp = self._trt_infer(x)
        elif self.model is not None:
            with torch.no_grad():
                prob_non_hsrp = torch.sigmoid(self.model(x).squeeze()).float().item()
        else:
            return self._empty()
        return self._result(prob_non_hsrp)

    def predict_batch(self, plate_images: list) -> list:
        return [self.predict(img) for img in plate_images]

    def _result(self, prob_non_hsrp: float) -> dict:
        prob_hsrp = 1.0 - prob_non_hsrp
        if prob_non_hsrp >= self.threshold:
            label, conf = "non_hsrp", prob_non_hsrp
        else:
            label, conf = "hsrp", prob_hsrp
        return {
            "label":          label,
            "confidence":     round(conf, 4),
            "prob_non_hsrp":  round(prob_non_hsrp, 4),
            "prob_hsrp":      round(prob_hsrp, 4),
        }

    def _empty(self):
        return {"label": None, "confidence": 0.0, "prob_non_hsrp": 0.0, "prob_hsrp": 0.0}