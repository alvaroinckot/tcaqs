#!/usr/bin/env python3
print("Starting minimal test...")

import gradio as gr
print("Gradio imported")

def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# Test App")
        btn = gr.Button("Test")
    return demo

print("Creating demo...")
demo = create_demo()
print("Demo created:", type(demo))

app = demo
print("App assigned:", type(app))

print("Script completed successfully")
