# 🧠 Neuro — Real-Time AI Brain with 3D Visualization

![Status](https://img.shields.io/badge/status-active-success)
![Python](https://img.shields.io/badge/python-3.10+-blue)

A living, learning neuromorphic brain that runs locally on your computer.
Watch it grow, learn, and form associations in real-time through a stunning 3D visualization.

---

## 🚀 Preview

> Real-time 3D brain + AI chat + learning system

//Preview//

---

## 🧠 What Is This?

NeuroLinked is a biologically-inspired artificial brain built with **100,000+ spiking neurons** across 11 brain regions.

It uses real neuroscience principles:

* **Spiking neurons** (Izhikevich model)
* **STDP learning** — “neurons that fire together, wire together”
* **11 brain regions** — Sensory, Association, Hippocampus, Prefrontal, Motor, etc.
* **Neuromodulators** — dopamine, acetylcholine, norepinephrine, serotonin
* **Memory consolidation**
* **Development stages** — EMBRYONIC → MATURE

---

## ⚡ Features

* 🧠 Real-time neural simulation
* 🌐 WebSocket-based AI chat
* 🤖 Local AI using **Ollama (LLaMA3)**
* 🎮 Interactive 3D brain visualization (Three.js)
* 🔄 Live learning & memory formation
* 💬 Chat interface with typing effect

---

## 🧠 AI Integration

NeuroLinked includes a local AI "brain" powered by **Ollama**.

* Runs fully offline
* Uses models like `llama3`
* Connected via WebSocket
* Responses appear in real-time in UI

### Run Ollama:

```bash
ollama run llama3
```

---

## ⚙️ Tech Stack

* **Backend**: FastAPI (Python)
* **AI**: Ollama (LLaMA3)
* **Frontend**: Vanilla JS
* **Visualization**: Three.js (3D brain)
* **Protocol**: WebSockets + MCP (Claude)

---

## 🚀 Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Run Server

```bash
python server.py
```

### 3. Open Dashboard

```
http://localhost:8000
```

---

## 🧪 Alternative (Windows)

```bash
install.bat
start.bat
```

---

## 🧠 How It Works

1. User input → encoded into neural signals
2. Sensory Cortex processes input
3. Association layers form connections
4. Hippocampus stores patterns
5. STDP strengthens pathways
6. Brain evolves over time

---

## 🧩 Project Structure

```
NeuroLinked/
  brain/           # Neural engine
  sensory/         # Encoders
  dashboard/       # 3D UI
  brain_state/     # Saved brain data
  server.py        # FastAPI server
  websocket.js     # WebSocket client
```

---

## 💾 Persistence

* Auto-saves every 5 minutes
* Loads previous brain state automatically
* Manual save via UI

---

## 🛠 Troubleshooting

**No response in UI?**

* Make sure Ollama is running
* Check WebSocket connection in console

**Brain not loading?**

* Wait 10–30 seconds
* Refresh browser

**High CPU?**

```bash
python run.py --neurons 25000
```

---

## 📌 Requirements

* Python 3.10+
* 4GB RAM (8GB recommended)
* Modern browser

---

## 🚀 Future Plans

* 🧠 Memory graph (visual neurons)
* ⚡ Real-time streaming responses
* 🤖 Agent-based reasoning
* 🎨 ChatGPT-style UI

---

## ⭐ Contributing

Feel free to fork, experiment, and build on top of NeuroLinked.

---

## 🧠 Vision

> Not just AI… a system that **learns, evolves, and visualizes thought.**

## 🚀 Future Vision & Capabilities

Neuro is evolving beyond a simple AI chat into a **true neuromorphic cognitive system**.

The original architecture supports far more advanced capabilities, including:

### 🧠 Advanced Brain Features

* Multi-region brain simulation (100K+ neurons across 11 regions)
* Spiking neural networks (Izhikevich model)
* STDP learning (self-improving synaptic connections)
* Memory consolidation via hippocampus replay
* Developmental stages (Embryonic → Mature brain)

---

### 🤖 AI & Cognitive Layer

* Integration with external AI systems (e.g., Claude via MCP)
* Hybrid intelligence (local + cloud reasoning)
* Context-aware responses based on learned patterns
* Agent-style behavior (planning, reasoning, decision-making)

---

### 👁️ Multi-Modal Learning

* Text input (current)
* Screen observation (learning from visual data)
* Audio input (microphone)
* Webcam perception

---

### 🌐 Real-Time Learning System

* Continuous learning from user interaction
* Automatic pattern grouping & concept formation
* Long-term memory building across sessions
* Adaptive behavior based on repeated inputs

---

### 🧩 Intended Purpose

This project was originally built to explore:

* How a **real artificial brain** could form memories and associations
* How AI can move beyond static responses → **dynamic learning systems**
* Visualizing **thought processes in real-time**
* Bridging neuroscience + AI + interactive systems

---

### 🚀 Planned Upgrades

* 🧠 Memory graph (messages become neurons)
* ⚡ True streaming responses (token-level)
* 🤖 Autonomous AI agent behavior
* 🎨 Advanced ChatGPT-style UI
* 🌍 Cross-platform support (Linux/Mac)
* 🧠 Deeper brain-region simulation accuracy

---

> ⚠️ Note: The current version focuses on a stable local AI system using Ollama.
> The full neuromorphic and multi-modal capabilities are part of the evolving roadmap.
