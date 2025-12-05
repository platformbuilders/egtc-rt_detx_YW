# ovd.py
import torch
from typing import List, Dict, Tuple
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class OpenVocabPPE:
    """
    Open-vocabulary detector (OWL-V2/OwlViT) para EPI em crops ou frame inteiro.
    Usa AutoProcessor/AutoModelForZeroShotObjectDetection para evitar mismatch de classe.
    """

class OpenVocabPPE:
    def __init__(
        self,
        model_name: str = "google/owlv2-base-patch16",
        device: str = "cuda",
        fp16: bool = False,
        cache_dir: str | None = None,
        use_fast: bool = True,
        quantization_mode: str = "none",   # "none" | "8bit" | "4bit"
        **kwargs,                           # ignora extras inesperados
    ):
        self.device = device
        self.fp16 = fp16
        self.quantization_mode = (quantization_mode or "none").lower()
        common = {"cache_dir": cache_dir} if cache_dir else {}

        # Processor
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, use_fast=use_fast, **common)
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(model_name, **common)

        # Modelo (com suporte opcional a bitsandbytes)
        load_kwargs = {}
        quantized = False
        if self.quantization_mode in ("8bit", "int8", "bnb-int8"):
            try:
                import bitsandbytes as bnb  # noqa: F401
                load_kwargs.update({"device_map": "auto", "load_in_8bit": True})
                quantized = True
            except Exception:
                quantized = False
        elif self.quantization_mode in ("4bit", "int4", "bnb-int4"):
            try:
                import bitsandbytes as bnb  # noqa: F401
                load_kwargs.update({
                    "device_map": "auto",
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16,
                    "bnb_4bit_quant_type": "nf4",
                })
                quantized = True
            except Exception:
                quantized = False

        try:
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_name, **common, **load_kwargs
            )
        except ValueError as e:
            if "state dictionary" in str(e).lower():
                self.processor = AutoProcessor.from_pretrained(model_name, force_download=True, **common)
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    model_name, force_download=True, **common, **load_kwargs
                )
            else:
                raise

        # Movimenta p/ device quando NÃO está quantizado (em bnb, device_map="auto" já cuida)
        if not quantized:
            self.model = self.model.to(device)

        self.model.eval()

        # fp16 apenas quando não está quantizado
        if fp16 and not quantized:
            try:
                self.model = self.model.half()
            except Exception:
                self.fp16 = False


    # ------------------------ internos ------------------------

    def _post(
        self,
        outputs,
        queries: List[str],
        sizes_hw: List[Tuple[int, int]],
        score_thr: float,
    ):
        """
        Converte a saída do modelo para dicts por rótulo:
        retorna (flags_list_placeholder, raw_list) onde
        raw_list[i] = {label_text: [(score, [x1,y1,x2,y2]), ...], ...}
        """
        target_sizes = torch.tensor([[h, w] for (h, w) in sizes_hw], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=score_thr
        )

        flags_list, raw_list = [], []
        for res in results:
            raw = {}
            for score, label_idx, box in zip(
                res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()
            ):
                label_text = queries[label_idx]
                raw.setdefault(label_text, []).append((float(score), [float(x) for x in box]))
            raw_list.append(raw)
            flags_list.append(raw)  # placeholder; consolidação acontece em infer()/infer_batch()
        return flags_list, raw_list

    # ------------------------ públicos ------------------------

    def infer(
        self,
        img_bgr: np.ndarray,
        positive: Dict[str, List[str]],
        negative: Dict[str, List[str]],
        score_thr: float = 0.24,
    ):
        """
        Inferência em uma única imagem/crop BGR.
        Retorna (flags_por_grupo, raw_por_label_text).
        """
        h, w = img_bgr.shape[:2]
        image = Image.fromarray(img_bgr[:, :, ::-1])  # BGR->RGB

        # prompt efetivo = concat(positive) + concat(negative)
        labels = [s for _, syns in positive.items() for s in syns]
        neg_labels = [s for _, syns in negative.items() for s in syns]
        queries = labels + neg_labels

        inputs = self.processor(text=[queries], images=image, return_tensors="pt")
        if self.fp16:
            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                    inputs[k] = v.half()
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        _, raw_list = self._post(outputs, queries, [(h, w)], score_thr)
        raw = raw_list[0]

        # Consolidação por grupo
        flags = {g: any(s in raw for s in syns) for g, syns in positive.items()}
        # Exemplo de override negativo específico (se existir no YAML)
        if ("helmet" in flags) and not flags["helmet"]:
            if any(s in raw for s in negative.get("non_helmet", [])):
                flags["helmet"] = False

        return flags, raw

    def infer_batch(
        self,
        crops_bgr: List[np.ndarray],
        positive: Dict[str, List[str]],
        negative: Dict[str, List[str]],
        score_thr: float = 0.24,
    ):
        """
        Inferência batelada em vários crops (BGR).
        Retorna (lista_de_flags_por_grupo, lista_de_raw_por_label_text).
        """
        if not crops_bgr:
            return [], []

        images = [Image.fromarray(c[:, :, ::-1]) for c in crops_bgr]  # BGR->RGB
        sizes_hw = [(img.size[1], img.size[0]) for img in images]  # (h, w)

        labels = [s for _, syns in positive.items() for s in syns]
        neg_labels = [s for _, syns in negative.items() for s in syns]
        queries = labels + neg_labels

        inputs = self.processor(text=[queries] * len(images), images=images, return_tensors="pt", padding=True)
        if self.fp16:
            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                    inputs[k] = v.half()
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        _, raw_list = self._post(outputs, queries, sizes_hw, score_thr)

        # Consolidação por grupo para cada crop
        flags_list = []
        for raw in raw_list:
            flags = {g: any(s in raw for s in syns) for g, syns in positive.items()}
            if ("helmet" in flags) and not flags["helmet"]:
                if any(s in raw for s in negative.get("non_helmet", [])):
                    flags["helmet"] = False
            flags_list.append(flags)

        return flags_list, raw_list
