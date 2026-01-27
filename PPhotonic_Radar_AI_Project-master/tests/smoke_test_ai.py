import torch
import numpy as np
from ai_models.dataset_generator import RadarDatasetGenerator
from ai_models.architectures import HybridRadarNet
from ai_models.xai import GradCAM, mc_dropout_inference, generate_explanation

def smoke_test_ai_suite():
    print("🚀 Starting Radar AI Suite Smoke Test...")
    
    # 1. Dataset Generation
    cfg = {"duration": 0.05, "fs": 5e5}
    gen = RadarDatasetGenerator(cfg)
    print("Generating synthetic sample: 'drone'...")
    sample = gen.generate_sample("drone")
    
    # 2. Model Initialization
    # Convert numpy and add batch dimension
    spec_tensor = torch.tensor(sample["rd_map"], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    ts_tensor = torch.tensor(sample["time_series"], dtype=torch.float32).unsqueeze(0)
    
    model = HybridRadarNet(num_classes=5)
    model.eval()
    
    # 3. Inference & Attention Check
    print("Running Inference...")
    with torch.no_grad():
        logits, attn_weights = model(spec_tensor, ts_tensor)
    
    assert logits.shape == (1, 5)
    assert attn_weights.shape[1] == 2000 # seq_len (1000 Real + 1000 Imag)
    print(f"[PASS] Model Inference successful. Logits: {logits.detach().numpy()}")
    print(f"[PASS] Attention Weights extracted. Shape: {attn_weights.shape}")

    # 4. XAI: Monte Carlo Dropout
    print("Running MC-Dropout Inference...")
    mean_probs, std_probs, entropy = mc_dropout_inference(model, spec_tensor, ts_tensor, n_iterations=5)
    print(f"[PASS] MC-Dropout successful. Entropy: {entropy:.4f}")

    # 5. XAI: Grad-CAM
    print("Running Grad-CAM...")
    # Target the last residual layer of the CNN branch
    cam_generator = GradCAM(model, model.cnn.layer3)
    cam = cam_generator.generate(spec_tensor, ts_tensor, class_idx=0)
    print(f"Grad-CAM shape: {cam.shape}")
    assert cam.shape == (32, 32)
    print(f"[PASS] Grad-CAM generated. Shape: {cam.shape}")

    # 6. Explanation Generation
    print("Generating Narrative Explanation...")
    expl = generate_explanation(
        prediction_class="drone",
        calibrated_probs=mean_probs,
        uncertainty_dict={"entropy": entropy},
        features={"snr_db": 20.0},
        cam_heatmap=cam
    )
    print(f"\n--- AI NARRATIVE ---\n{expl.narrative}\n")
    print(f"Confidence: {expl.calibrated_confidence:.4f}")
    assert expl.verification_passed == True
    print("[PASS] Explanation logic verified.")

if __name__ == "__main__":
    try:
        smoke_test_ai_suite()
        print("\n✅ ALL AI SUITE SMOKE TESTS PASSED!")
    except Exception as e:
        print(f"\n❌ SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
