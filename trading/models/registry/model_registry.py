"""
Model Registry: Manages versions, metadata, and artifacts of production ML models.
Ensures every prediction and trade references an explicit model_version.
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List


class ModelRegistry:
    def __init__(self, registry_dir: Optional[str] = None):
        if registry_dir is None:
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            registry_dir = os.path.join(root, "trading", "models", "registry", "store")
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)
        self.metadata_file = os.path.join(self.registry_dir, "models_metadata.json")
        self._ensure_metadata_file()

    def _ensure_metadata_file(self):
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, "w") as f:
                json.dump({"active_version": None, "models": {}}, f, indent=2)

    def _load_metadata(self) -> Dict[str, Any]:
        with open(self.metadata_file, "r") as f:
            return json.load(f)

    def _save_metadata(self, data: Dict[str, Any]):
        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=2)

    def register_model(
        self,
        algorithm_name: str,
        artifact_path: str,
        metrics: Dict[str, Any],
        features: List[str],
        version_id: Optional[str] = None
    ) -> str:
        """Register a newly trained model version."""
        if not version_id:
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            short_hash = hashlib.md5(f"{algorithm_name}_{now_str}".encode()).hexdigest()[:6]
            version_id = f"{algorithm_name.lower()}_v{now_str}_{short_hash}"

        data = self._load_metadata()
        data["models"][version_id] = {
            "version_id": version_id,
            "algorithm_name": algorithm_name,
            "artifact_path": artifact_path,
            "registered_at": datetime.now().isoformat(),
            "metrics": metrics,
            "feature_count": len(features),
            "features": features,
            "is_active": False
        }

        # If no active model, make this active
        if data.get("active_version") is None:
            data["active_version"] = version_id
            data["models"][version_id]["is_active"] = True

        self._save_metadata(data)
        return version_id

    def set_active_version(self, version_id: str) -> bool:
        data = self._load_metadata()
        if version_id not in data["models"]:
            return False
        for vid in data["models"]:
            data["models"][vid]["is_active"] = (vid == version_id)
        data["active_version"] = version_id
        self._save_metadata(data)
        return True

    def get_active_model_info(self) -> Optional[Dict[str, Any]]:
        data = self._load_metadata()
        active_vid = data.get("active_version")
        if not active_vid or active_vid not in data["models"]:
            return None
        return data["models"][active_vid]

    def list_models(self) -> List[Dict[str, Any]]:
        data = self._load_metadata()
        return list(data["models"].values())
