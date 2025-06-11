#!/usr/bin/env python3
"""
Test version to isolate the issue
"""

import gradio as gr

print("Starting test app...")

# Create a simple Gradio interface
with gr.Blocks(title="Test App") as demo:
    gr.Markdown("# Test Application")
    
    with gr.Row():
        input_text = gr.Textbox(label="Input")
        output_text = gr.Textbox(label="Output")
    
    def process_text(text):
        return f"Processed: {text}"
    
    input_text.change(fn=process_text, inputs=input_text, outputs=output_text)

print("Demo created successfully")

# Make available at module level
app = demo
iface = demo

print("App assigned successfully")
print("App type:", type(app))

if __name__ == "__main__":
    print("Running main...")
    demo.launch(server_port=7860, share=False)
