#!/usr/bin/env python3
"""
Minimal test version
"""

print("🚀 Starting minimal test...")

import gradio as gr

print("📦 Gradio imported successfully")

def test_predict(text):
    return f"You entered: {text}"

print("🔧 Creating interface...")

demo = gr.Interface(
    fn=test_predict,
    inputs="text",
    outputs="text",
    title="Test App"
)

print("✅ Interface created")

app = demo
print(f"✅ App exported: {type(app)}")

if __name__ == "__main__":
    print("🌐 Starting server...")
    demo.launch(server_port=7860)
