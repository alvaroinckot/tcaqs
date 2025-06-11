#!/usr/bin/env python3
print("Test 1: Script starting")

import gradio as gr
print("Test 2: Gradio imported")

demo = gr.Interface(fn=lambda x: x, inputs="text", outputs="text")
print("Test 3: Interface created")

app = demo
print("Test 4: App assigned")

print("Test 5: Script ending")
