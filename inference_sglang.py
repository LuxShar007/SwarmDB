import sglang as sg
import time

# Define a high-speed structured generation graph
@sg.function
def optimize_telemetry_extraction(s, telemetry_dump):
    # This system instruction tree gets permanently frozen in VRAM via RadixAttention
    s += "You are the SwarmDB AI Core. Convert unstructured logs into rigid json arrays.\n"
    s += f"Log Data: {telemetry_dump}\n"
    s += "Output JSON: "
    
    # Force the model to skip conversational chat and output ONLY numbers
    s += sg.gen(
        "json_output",
        max_tokens=64,
        temperature=0.0,
        regex=r'\{\"id\": [0-9]+, \"vector\": \[-?[0-9]+\.[0-9]+, -?[0-9]+\.[0-9]+, -?[0-9]+\.[0-9]+\]\}'
    )

def test_inference_pipeline():
    raw_incoming_text = "ALERT: Drone 77 lost stabilization matrix, recovering at location X=12.45, Y=-98.21, Z=4.50."
    
    print("[*] Binding to Saumitra's SGLang Backend Runtime (Port: 30000)...")
    # Pointing to the high-speed local inference cluster endpoint
    sg.set_default_backend(sg.RuntimeEndpoint("http://localhost:30000"))
    
    print("[*] Launching accelerated prefix-cached inference call...")
    start_time = time.time()
    
    state = optimize_telemetry_extraction.run(telemetry_dump=raw_incoming_text)
    
    end_time = time.time()
    
    print("\n=== SGLang High-Speed Inference Metrics ===")
    print(f"| Raw Text Processed:  {raw_incoming_text}")
    print(f"| Optimized JSON Out:  {state['json_output']}")
    print(f"| Inference Latency:   {end_time - start_time:.6f} seconds")
    print("===========================================\n")

if __name__ == "__main__":
    print("[!] Run this after launching: python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct")
# test_inference_pipeline() # Uncomment this once the local backend engine server is active!