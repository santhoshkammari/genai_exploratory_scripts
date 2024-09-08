# Spotlight LLM (SIMILAR TO MAC SPOTLIGHT)

Spotlight LLM is a Python application that provides a quick access interface to interact with local language models using Ollama.

## Prerequisites

- Python 3.x
- PyQt5
- Langchain
- Ollama

## Installation

1. Clone this repository or download the `spotlight_llm.py` script.
2. Install the required Python packages:

```bash
pip install PyQt5 langchain-community
```

3. Install Ollama following the instructions at [https://ollama.ai/](https://ollama.ai/)

## Usage

Run the script using Python:

```bash
python3 /path/to/spotlight_llm.py
```

## Setting up a GNOME shortcut

To set up a keyboard shortcut (Ctrl+Alt+Space) to launch Spotlight LLM in GNOME:

1. Open Settings
2. Go to Keyboard Shortcuts
3. Click the '+' at the bottom to add a new shortcut
4. Set the command to: `/usr/bin/python3 /path/to/spotlight_llm.py`
5. Set the shortcut to Ctrl+Alt+Space

## Features

- Quick access interface for interacting with local language models
- Multiple model support (qwen2:0.5b, qwen2:0.5b-instruct, gemma2:2b)
- Transparent, always-on-top window
- Saves interaction history

## Contributing

Feel free to open issues or submit pull requests to improve Spotlight LLM.

## License

[Specify your license here]
