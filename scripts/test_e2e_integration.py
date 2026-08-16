"""
scripts/test_e2e_integration.py
Real end-to-end test of Day 5 pipeline:
SAR image -> FastAPI POST /api/colorize -> SSG-U-Net checkpoint -> RGB prediction -> MC-Dropout -> Trust-gated output -> SQLite persistence.
"""
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure runtime directories exist
os.environ.setdefault("RUNTIME_DIR", "runtime")
os.environ.setdefault("UPLOAD_DIR", "runtime/uploads")
os.environ.setdefault("RESULTS_DIR", "runtime/results")
os.environ.setdefault("DATABASE_URL", "sqlite:///./runtime/app.db")
os.environ.setdefault("MODEL_CHECKPOINT", "runtime/checkpoints/best_model.pt")

from fastapi.testclient import TestClient
from backend.app.main import app

def run_e2e_test():
    sar_image_path = Path("data/raw/sih1733/Pair-1/SAR-Image-1.jpg")
    print(f"[E2E Test] Testing with SAR image: {sar_image_path}")
    assert sar_image_path.exists(), f"SAR image not found: {sar_image_path}"

    with TestClient(app) as client:
        # 1. Health check
        print("[E2E Test] 1. Checking /api/health...")
        r_health = client.get("/api/health")
        assert r_health.status_code == 200, f"Health check failed: {r_health.text}"
        health_data = r_health.json()
        print(f"   -> Health response: {health_data}")
        assert health_data["status"] == "ok", "Expected status ok"
        assert health_data["checkpoint_exists"] is True, "Expected checkpoint_exists True"

        # 2. Upload & Colorize
        print("[E2E Test] 2. Uploading SAR image to POST /api/colorize...")
        with open(sar_image_path, "rb") as f:
            r_colorize = client.post(
                "/api/colorize",
                files={"file": ("SAR-Image-1.jpg", f, "image/jpeg")},
            )
        assert r_colorize.status_code in (200, 202), f"Colorize request failed with status {r_colorize.status_code}: {r_colorize.text}"
        res_data = r_colorize.json()
        job_id = res_data["id"]
        print(f"   -> Created Job ID: {job_id}, Initial Status: {res_data['status']}")

        # 3. Poll /api/colorize/{id} until done
        print(f"[E2E Test] 3. Polling GET /api/colorize/{job_id} for completion...")
        max_retries = 20
        job_data = None
        for i in range(max_retries):
            r_job = client.get(f"/api/colorize/{job_id}")
            assert r_job.status_code == 200, f"GET job failed: {r_job.text}"
            job_data = r_job.json()
            if job_data["status"] in ("done", "error"):
                break
            time.sleep(0.5)

        print(f"   -> Final Job Status: {job_data['status']}")
        assert job_data["status"] == "done", f"Job failed with error: {job_data.get('error_message')}"

        trust_score = job_data.get("uncertainty_mean")
        result_url = job_data.get("result_url")
        uncertainty_url = job_data.get("uncertainty_url")

        print(f"   -> Mean Uncertainty: {trust_score}")
        print(f"   -> Result URL: {result_url}")
        print(f"   -> Uncertainty URL: {uncertainty_url}")
        assert result_url is not None, "Result URL is missing"
        assert uncertainty_url is not None, "Uncertainty URL is missing"

        # 4. Check generated result files on disk
        result_file = Path("runtime/results") / f"{job_id}_colorized.png"
        uncertainty_file = Path("runtime/results") / f"{job_id}_uncertainty.png"
        print(f"[E2E Test] 4. Verifying output files on disk...")
        print(f"   -> Result file: {result_file} (exists={result_file.exists()}, size={result_file.stat().st_size if result_file.exists() else 0} bytes)")
        print(f"   -> Uncertainty file: {uncertainty_file} (exists={uncertainty_file.exists()}, size={uncertainty_file.stat().st_size if uncertainty_file.exists() else 0} bytes)")

        assert result_file.exists(), f"Result file missing: {result_file}"
        assert uncertainty_file.exists(), f"Uncertainty file missing: {uncertainty_file}"
        assert result_file.stat().st_size > 0, "Result file is empty"
        assert uncertainty_file.stat().st_size > 0, "Uncertainty file is empty"

        # 5. GET /api/history
        print("[E2E Test] 5. Verifying analysis history via GET /api/history...")
        r_hist = client.get("/api/history")
        assert r_hist.status_code == 200, f"GET history failed: {r_hist.text}"
        hist_data = r_hist.json()
        print(f"   -> History total jobs: {hist_data['total']}")
        assert hist_data["total"] >= 1, "Expected at least 1 job in history"
        found_job = any(j["id"] == job_id for j in hist_data["jobs"])
        assert found_job, f"Job {job_id} not found in history list"

        print("\n==================================================")
        print("REAL END-TO-END TEST PASSED SUCCESSFULLY!")
        print("==================================================")

if __name__ == "__main__":
    run_e2e_test()
