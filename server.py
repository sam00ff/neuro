"""
FastAPI + WebSocket Server for NeuroLinked Brain

Serves the 3D dashboard and streams real-time brain state via WebSocket.
Provides Claude integration API for reading brain state and sending input.
"""
from ollama_chat import ask_ollama

import asyncio
import json
import os
import sys
import threading
import time

from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from brain.brain import Brain
from brain.config import BrainConfig
from brain.persistence import (
    save_brain, load_brain, get_save_info,
    list_backups, restore_backup, is_save_locked, get_lock_reason, unlock_save,
)
from brain.claude_bridge import ClaudeBridge
from brain.screen_observer import ScreenObserver
from brain.video_recorder import VideoRecorder
# --- V1.3: Agent execution layer ---
from brain.agent import build_default_agent, plan_from_json, PlanStep
from brain.tools import (
    describe_tools, call_tool, tool_history, tool_stats,
)
# --- V1.3: Full-duplex voice pipeline ---
from brain.voice import VoicePipeline, PipelineState
# --- V1.3: Multi-tenant brain management ---
from brain.tenant import tenant_manager
# --- V1.3: Enterprise security ---
from brain.security import audit_log, rate_limiter, redact_pii
# --- V1.3: Public benchmarks ---
from brain import benchmarks as bench_mod
# --- V1.3: Integration marketplace ---
from brain.plugins import discover_and_load as _load_plugins, list_plugins
# --- V1.3: Autonomous background thinking ---
from brain.sleep_consolidation import SleepConsolidator
# --- V1.3: Event bus + webhooks + SSE ---
from brain.events import event_bus, webhook_manager

from sensory.text import TextEncoder
from sensory.vision import VisionEncoder
from sensory.audio import AudioEncoder

app = FastAPI(title="NeuroLinked Brain", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
brain: Brain = None
text_encoder: TextEncoder = None
vision_encoder: VisionEncoder = None
audio_encoder: AudioEncoder = None
claude_bridge: ClaudeBridge = None
screen_observer: ScreenObserver = None
video_recorder: VideoRecorder = None
# V1.3 agent (wired to knowledge store in init_brain)
agent = None
# V1.3 voice pipeline (lazy-init on first use)
voice_pipeline: VoicePipeline = None
# V1.3 sleep consolidator (started in init_brain)
sleep_consolidator: SleepConsolidator = None

# Simulation thread control
sim_running = False
sim_thread = None
connected_clients = set()

# Auto-save interval (every 5 minutes)
AUTO_SAVE_INTERVAL = 300
_last_auto_save = 0


def init_brain():
    """Initialize the brain and sensory encoders."""
    global brain, text_encoder, vision_encoder, audio_encoder, claude_bridge, screen_observer, video_recorder, agent, sleep_consolidator
    brain = Brain()
    text_encoder = TextEncoder(feature_dim=256)
    vision_encoder = VisionEncoder(feature_dim=256)
    audio_encoder = AudioEncoder(feature_dim=256)
    claude_bridge = ClaudeBridge(brain)
    screen_observer = ScreenObserver(feature_dim=256, capture_interval=2.0)
    # Wire up screen observer so OCR text flows into brain + knowledge store
    screen_observer.attach_brain(
        brain=brain,
        text_encoder=text_encoder,
        knowledge_store=claude_bridge.knowledge,
    )
    # Video recorder saves screen to .mp4 segments (off by default)
    video_recorder = VideoRecorder(fps=10, segment_minutes=10)

    # V1.3: auto-discover plugins BEFORE the agent is built so plugin tools
    # are registered into the tool registry from the start.
    try:
        loaded_plugins = _load_plugins()
        if loaded_plugins:
            print(f"[SERVER] Loaded {len(loaded_plugins)} plugin(s): "
                  f"{[p.name for p in loaded_plugins]}")
    except Exception as e:
        print(f"[SERVER] Plugin discovery failed (non-fatal): {e}")

    # V1.3: agent execution layer — bound to the brain's knowledge store
    agent = build_default_agent()
    # 🔥 CONNECT OLLAMA (THIS IS THE KEY LINE)
    if agent is not None and claude_bridge is not None:
        agent.knowledge_store = claude_bridge.knowledge

    print("[SERVER] Agent ready — tools available:",
        [t["name"] for t in describe_tools()])


    # V1.3: autonomous background thinking — starts a thread that consolidates
    # memory during idle periods. Completely optional; non-fatal if it fails.
    try:
        if claude_bridge is not None and claude_bridge.knowledge is not None:
            sleep_consolidator = SleepConsolidator(
                knowledge_store=claude_bridge.knowledge,
                idle_threshold_sec=300.0,
                hourly_tick=True,
            )
            sleep_consolidator.start()
            print("[SERVER] Sleep consolidator running")
    except Exception as e:
        print(f"[SERVER] Sleep consolidator disabled (non-fatal): {e}")

    # Try to load saved state
    loaded = load_brain(brain)
    if loaded:
        print("[SERVER] Restored brain from saved state")
    else:
        print("[SERVER] Starting fresh brain")


_last_screen_log = 0
SCREEN_LOG_INTERVAL = 30  # Log screen activity to knowledge every 30 seconds

def simulation_loop():
    """Run brain simulation in background thread."""
    global sim_running, _last_auto_save, _last_screen_log
    target_dt = 1.0 / 100  # Target 100 steps/sec
    while sim_running:
        start = time.time()
        try:
            # Feed screen observation if active
            if screen_observer and screen_observer.active:
                features = screen_observer.get_features()
                brain.inject_sensory_input("vision", features)

                # Periodically log screen activity to knowledge store
                now = time.time()
                if now - _last_screen_log > SCREEN_LOG_INTERVAL and claude_bridge:
                    try:
                        screen_state = screen_observer.get_state()
                        motion = screen_state.get("motion", 0)
                        if motion > 0.01:  # Only log if there's actual screen activity
                            claude_bridge.knowledge.store(
                                text=f"Screen activity detected: motion level {motion:.1%}, "
                                     f"brain step {brain.step_count}",
                                source="screen_observer",
                                tags=["screen", "observation", "auto"],
                            )
                    except Exception:
                        pass
                    _last_screen_log = now

            brain.step()

            # Auto-save periodically
            now = time.time()
            if now - _last_auto_save > AUTO_SAVE_INTERVAL:
                try:
                    save_brain(brain)
                    _last_auto_save = now
                except Exception as e:
                    print(f"[SERVER] Auto-save error: {e}")

        except Exception as e:
            print(f"[SIM] Error: {e}")
        elapsed = time.time() - start
        sleep_time = target_dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def start_simulation():
    """Start the background simulation thread."""
    global sim_running, sim_thread
    if sim_running:
        return
    sim_running = True
    sim_thread = threading.Thread(target=simulation_loop, daemon=True)
    sim_thread.start()
    print("[SERVER] Simulation started")


def stop_simulation():
    """Stop the background simulation thread."""
    global sim_running
    sim_running = False
    print("[SERVER] Simulation stopped")


# --- Static files ---
# When frozen by PyInstaller, dashboard lives next to the .exe, not next to this file.
if getattr(sys, "frozen", False):
    _base_dir = os.path.dirname(sys.executable)
else:
    _base_dir = os.path.dirname(__file__)
dashboard_path = os.path.join(_base_dir, "dashboard")
# Fallback: if the user-editable dashboard folder is missing, look inside the bundle.
if not os.path.isdir(dashboard_path):
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")
app.mount("/css", StaticFiles(directory=os.path.join(dashboard_path, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(dashboard_path, "js")), name="js")


@app.on_event("startup")
async def startup():
    init_brain()
    start_simulation()


@app.on_event("shutdown")
async def shutdown():
    # Save brain state on shutdown
    try:
        save_brain(brain)
        print("[SERVER] Brain saved on shutdown")
    except Exception as e:
        print(f"[SERVER] Save on shutdown failed: {e}")

    stop_simulation()
    if screen_observer:
        screen_observer.stop()
    if video_recorder:
        video_recorder.stop()
    if vision_encoder:
        vision_encoder.stop_webcam()
    if audio_encoder:
        audio_encoder.stop_microphone()


# --- Routes ---

@app.get("/")
async def index():
    return FileResponse(os.path.join(dashboard_path, "index.html"))


@app.get("/api/state")
async def get_state():
    return JSONResponse(brain.get_state())


@app.get("/api/positions")
async def get_positions():
    return JSONResponse(brain.get_neuron_positions())


@app.post("/api/input/text")
async def input_text(data: dict):
    text = data.get("text", "")

    if not text:
        return {"status": "error", "message": "No text provided"}

    # 🧠 keep memory injection
    features = text_encoder.encode(text)
    brain.inject_sensory_input("text", features)

    if claude_bridge:
        claude_bridge.send_observation({
            "type": "text",
            "content": text,
            "source": "user",
        })

    # 🔥 THIS IS THE REAL AI PART
    try:
        from ollama_chat import ask_ollama

        result = ask_ollama(text)
    except Exception as e:
        return {"status": "error", "message": str(e)}

    return {
        "status": "ok",
        "response": result
    }


@app.post("/api/input/vision/start")
async def start_vision():
    success = vision_encoder.start_webcam()
    return {"status": "started" if success else "unavailable"}


@app.post("/api/input/vision/stop")
async def stop_vision():
    vision_encoder.stop_webcam()
    return {"status": "stopped"}


@app.post("/api/input/audio/start")
async def start_audio():
    success = audio_encoder.start_microphone()
    return {"status": "started" if success else "unavailable"}


@app.post("/api/input/audio/stop")
async def stop_audio():
    audio_encoder.stop_microphone()
    return {"status": "stopped"}


@app.post("/api/control/pause")
async def pause():
    stop_simulation()
    return {"status": "paused"}


@app.post("/api/control/resume")
async def resume():
    start_simulation()
    return {"status": "running"}


@app.post("/api/control/reset")
async def reset():
    stop_simulation()
    init_brain()
    start_simulation()
    return {"status": "reset"}


# =============================================================================
# Claude Integration API
# =============================================================================

@app.get("/api/claude/summary")
async def claude_summary():
    """Primary endpoint for Claude to read brain state."""
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    try:
        result = claude_bridge.get_brain_summary()
        return JSONResponse(json.loads(json.dumps(result, default=str)))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/claude/insights")
async def claude_insights():
    """Get brain-derived insights useful for Claude."""
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    try:
        result = claude_bridge.get_insights()
        return JSONResponse(json.loads(json.dumps(result, default=str)))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/claude/observe")
async def claude_observe(data: dict):
    """
    Claude sends an observation to the brain.
    Body: {"type": "text"|"action"|"context", "content": "...", "source": "claude"}
    """
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    claude_bridge.send_observation(data)
    return {"status": "ok", "interaction_count": claude_bridge._interaction_count}


@app.get("/api/claude/status")
async def claude_status():
    """Get Claude bridge connection status."""
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    state = claude_bridge.get_state()
    if screen_observer:
        state["screen_observer"] = screen_observer.get_state()
    if video_recorder:
        state["video_recorder"] = video_recorder.get_state()
    return JSONResponse(state)


@app.get("/api/claude/activity")
async def claude_activity():
    """Get recent activity log."""
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    return JSONResponse(claude_bridge.get_activity_log())


@app.get("/api/claude/learned")
async def claude_learned():
    """Get what the brain has learned - grouped patterns and associations."""
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    try:
        result = claude_bridge.get_learned_patterns()
        return JSONResponse(json.loads(json.dumps(result, default=str)))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/claude/learned/summary")
async def claude_learned_summary():
    """Get plain-English summary of what the brain has learned."""
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    try:
        text = claude_bridge.get_learning_summary()
        return JSONResponse({"summary": text})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================================
# Knowledge Store API (text storage & retrieval — replaces Obsidian)
# =============================================================================

@app.get("/api/claude/recall")
async def claude_recall(q: str = "", limit: int = 10):
    """Recall knowledge about a specific topic."""
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    if not q:
        return JSONResponse({"error": "Query parameter 'q' is required"}, status_code=400)
    try:
        results = claude_bridge.recall(q, limit=limit)
        return JSONResponse({"query": q, "results": results, "count": len(results)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/claude/search")
async def claude_search(q: str = "", limit: int = 20):
    """Full-text search across all stored knowledge."""
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    if not q:
        return JSONResponse({"error": "Query parameter 'q' is required"}, status_code=400)
    try:
        results = claude_bridge.search_knowledge(q, limit=limit)
        return JSONResponse({"query": q, "results": results, "count": len(results)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/claude/semantic")
async def claude_semantic(q: str = "", limit: int = 10):
    """Semantic (associative) search - finds conceptually related memories
    via TF-IDF cosine similarity, not just keyword matching."""
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    if not q:
        return JSONResponse({"error": "Query parameter 'q' is required"}, status_code=400)
    try:
        results = claude_bridge.knowledge.semantic_search(q, limit=limit)
        return JSONResponse({"query": q, "results": results, "count": len(results),
                             "mode": "semantic_tfidf"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/claude/knowledge")
async def claude_knowledge():
    """Get knowledge store stats and recent entries."""
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    try:
        stats = claude_bridge.get_knowledge_stats()
        recent = claude_bridge.get_recent_knowledge(limit=10)
        return JSONResponse({"stats": stats, "recent": recent})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/claude/remember")
async def claude_remember(data: dict):
    """
    Store a piece of knowledge directly.
    Body: {"text": "...", "source": "claude", "tags": ["optional", "tags"]}
    """
    if not claude_bridge:
        return JSONResponse({"error": "Bridge not initialized"}, status_code=503)
    text = data.get("text", "")
    if not text:
        return JSONResponse({"error": "text field is required"}, status_code=400)
    source = data.get("source", "claude")
    tags = data.get("tags", None)
    try:
        entry_id = claude_bridge.store_knowledge(text=text, source=source, tags=tags)
        return {"status": "stored", "id": entry_id}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================================
# Screen Observation API
# =============================================================================

@app.post("/api/screen/start")
async def start_screen():
    """Start screen observation."""
    if not screen_observer:
        return JSONResponse({"error": "Screen observer not initialized"}, status_code=503)
    success = screen_observer.start()
    return {"status": "started" if success else "unavailable"}


@app.post("/api/screen/stop")
async def stop_screen():
    """Stop screen observation."""
    if screen_observer:
        screen_observer.stop()
    return {"status": "stopped"}


@app.get("/api/screen/state")
async def screen_state():
    """Get screen observer state."""
    if not screen_observer:
        return JSONResponse({"error": "Screen observer not initialized"}, status_code=503)
    return JSONResponse(screen_observer.get_state())


# =============================================================================
# Video Recording API
# =============================================================================

@app.post("/api/video/start")
async def start_video():
    """Start video recording (saves screen to .mp4 segments)."""
    if not video_recorder:
        return JSONResponse({"error": "Video recorder not initialized"}, status_code=503)
    success = video_recorder.start()
    return {"status": "started" if success else "unavailable"}


@app.post("/api/video/stop")
async def stop_video():
    """Stop video recording and close current segment."""
    if video_recorder:
        video_recorder.stop()
    return {"status": "stopped"}


@app.get("/api/video/state")
async def video_state():
    """Get video recorder state (active, fps, disk usage, file count)."""
    if not video_recorder:
        return JSONResponse({"error": "Video recorder not initialized"}, status_code=503)
    return JSONResponse(video_recorder.get_state())


@app.get("/api/video/list")
async def video_list():
    """List all recorded .mp4 files with size and timestamps."""
    if not video_recorder:
        return JSONResponse({"error": "Video recorder not initialized"}, status_code=503)
    return JSONResponse({"recordings": video_recorder.list_recordings()})


@app.post("/api/video/delete")
async def video_delete(data: dict):
    """Delete a recording by filename. Body: {'name': 'screen_YYYYMMDD_HHMMSS.mp4'}"""
    if not video_recorder:
        return JSONResponse({"error": "Video recorder not initialized"}, status_code=503)
    name = data.get("name")
    if not name:
        return JSONResponse({"error": "Missing 'name' field"}, status_code=400)
    success = video_recorder.delete_recording(name)
    return {"status": "deleted" if success else "not_found", "name": name}


@app.get("/api/video/recording/{filename}")
async def video_download(filename: str):
    """Stream/download a specific recording file."""
    if not video_recorder:
        return JSONResponse({"error": "Video recorder not initialized"}, status_code=503)
    # Only allow files in the recordings directory, and only .mp4
    if not filename.endswith(".mp4") or "/" in filename or "\\" in filename or ".." in filename:
        return JSONResponse({"error": "Invalid filename"}, status_code=400)
    path = os.path.join(video_recorder.output_dir, filename)
    if not os.path.isfile(path):
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(path, media_type="video/mp4", filename=filename)


# =============================================================================
# Persistence API
# =============================================================================

@app.post("/api/brain/save")
async def save_state():
    """Save brain state to disk."""
    try:
        save_brain(brain)
        return {"status": "saved", "step": brain.step_count}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/brain/load")
async def load_state():
    """Load brain state from disk."""
    try:
        stop_simulation()
        success = load_brain(brain)
        start_simulation()
        return {"status": "loaded" if success else "no_save_found", "step": brain.step_count}
    except Exception as e:
        start_simulation()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/brain/save-info")
async def save_info():
    """Get info about saved state without loading."""
    info = get_save_info()
    if info:
        return JSONResponse(info)
    return JSONResponse({"saved": False})


@app.get("/api/brain/backups")
async def brain_backups():
    """List all available brain state backups."""
    return JSONResponse({
        "backups": list_backups(),
        "save_locked": is_save_locked(),
        "lock_reason": get_lock_reason(),
    })


@app.post("/api/brain/restore-backup")
async def brain_restore_backup(data: dict):
    """Restore a specific backup. Body: {'name': 'backup_folder_name'}"""
    name = data.get("name", "")
    if not name:
        return JSONResponse({"error": "name field required"}, status_code=400)
    try:
        stop_simulation()
        success = restore_backup(name)
        if success:
            init_brain()
            start_simulation()
            return {"status": "restored", "backup": name, "step": brain.step_count}
        else:
            start_simulation()
            return JSONResponse({"error": "Backup not found"}, status_code=404)
    except Exception as e:
        start_simulation()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/brain/unlock")
async def brain_unlock(data: dict = None):
    """
    Unlock save protection. Required if neuron count mismatch locked saving.
    Body: {'confirm': true} - user must confirm they want to overwrite preserved state
    """
    data = data or {}
    if not data.get("confirm", False):
        return JSONResponse({
            "error": "Confirmation required",
            "message": "Pass {'confirm': true} to acknowledge you want to overwrite preserved state.",
            "lock_reason": get_lock_reason(),
        }, status_code=400)
    unlock_save(user_consent=True)
    return {"status": "unlocked", "warning": "Next save will overwrite preserved state"}


@app.get("/api/brain/lock-status")
async def brain_lock_status():
    """Check if save is currently locked."""
    return JSONResponse({
        "locked": is_save_locked(),
        "reason": get_lock_reason(),
    })


# =============================================================================
# V1.3 — Agent Execution Layer
# =============================================================================

@app.get("/api/agent/tools")
async def agent_list_tools():
    """List all tools the agent can call."""
    return {
        "tools": describe_tools(),
        "count": len(describe_tools()),
    }


@app.post("/api/agent/tool")
async def agent_call_single_tool(req: dict):
    """Invoke a single tool directly. req = {tool: str, args: dict}"""
    tool = req.get("tool")
    args = req.get("args", {}) or {}
    if not tool:
        return {"ok": False, "error": "Missing 'tool' in request"}
    result = call_tool(tool, args)
    return result.to_dict()


@app.post("/api/agent/plan")
async def agent_run_plan(req: dict):
    """
    Execute a plan (list of tool calls) in order.
    req = {
        goal: str,
        plan: [
            {tool: str, args: {...}, expect?: str, note?: str},
            ...
        ]
    }
    """
    if agent is None:
        return {"ok": False, "error": "Agent not initialized"}
    goal = req.get("goal", "")
    plan_raw = req.get("plan", [])
    if not isinstance(plan_raw, list) or not plan_raw:
        return {"ok": False, "error": "plan must be a non-empty list"}
    try:
        plan = plan_from_json(plan_raw)
    except Exception as e:
        return {"ok": False, "error": f"Invalid plan: {e}"}
    run = agent.run_plan(goal, plan)
    return run.to_dict()


@app.get("/api/agent/runs")
async def agent_recent_runs(limit: int = 20):
    """Recent agent runs."""
    if agent is None:
        return {"runs": []}
    return {"runs": agent.recent(limit=limit)}


@app.get("/api/agent/run/{run_id}")
async def agent_get_run(run_id: str):
    """Details of a specific run."""
    if agent is None:
        return {"ok": False, "error": "Agent not initialized"}
    run = agent.get_run(run_id)
    if run is None:
        return {"ok": False, "error": "run not found"}
    return run.to_dict()


@app.get("/api/agent/stats")
async def agent_stats_endpoint():
    """Aggregate agent + tool stats (feeds the benchmarks page)."""
    if agent is None:
        return {"ok": False, "error": "Agent not initialized"}
    return agent.stats()


@app.get("/api/agent/tool-history")
async def agent_tool_history(limit: int = 100):
    """Raw tool-call history (every call, every outcome)."""
    return {"history": tool_history(limit=limit)}


# =============================================================================
# V1.3 — Multi-Tenant Brain Management
# =============================================================================
# One install, many brains. Each employee gets their own isolated memory.
# Teams share knowledge via channels. Critical for business deployments.

from fastapi import Header, Query


def _extract_token(
    x_nl_tenant_token: Optional[str] = Header(None),
    token: Optional[str] = Query(None),
) -> Optional[str]:
    """Token can come via header OR ?token= query param (mobile-friendly)."""
    return x_nl_tenant_token or token


@app.post("/api/tenants")
async def tenant_create(req: dict):
    """
    Create a new tenant. Returns the tenant's token (only shown here).
    req = {"name": "Sarah - Marketing", "role": "user"|"admin"|"readonly", "metadata": {...}}
    """
    name = (req.get("name") or "").strip()
    if not name:
        return {"ok": False, "error": "name is required"}
    role = req.get("role", "user")
    try:
        t = tenant_manager.create_tenant(
            name=name, role=role, metadata=req.get("metadata", {}),
        )
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    # Return full dict ONCE at creation so the client can capture the token.
    return {"ok": True, "tenant": t.full_dict()}


@app.get("/api/tenants")
async def tenant_list():
    return {"tenants": [t.public_dict() for t in tenant_manager.list_tenants()]}


@app.get("/api/tenants/{tenant_id}")
async def tenant_get(tenant_id: str):
    t = tenant_manager.get_tenant(tenant_id)
    if t is None:
        return {"ok": False, "error": "tenant not found"}
    return t.public_dict()


@app.delete("/api/tenants/{tenant_id}")
async def tenant_delete(tenant_id: str, wipe_data: bool = False):
    ok = tenant_manager.delete_tenant(tenant_id, wipe_data=wipe_data)
    return {"ok": ok}


@app.post("/api/tenants/{tenant_id}/rotate-token")
async def tenant_rotate_token(tenant_id: str):
    token = tenant_manager.rotate_token(tenant_id)
    if token is None:
        return {"ok": False, "error": "tenant not found"}
    return {"ok": True, "new_token": token}


@app.post("/api/tenants/me/remember")
async def tenant_store(req: dict, token: Optional[str] = Depends(_extract_token)):
    """Store a memory into the caller's tenant knowledge store (by token)."""
    try:
        t = tenant_manager.require_role(token, "admin", "user")
    except PermissionError as e:
        return {"ok": False, "error": str(e)}
    text = (req.get("text") or "").strip()
    if not text:
        return {"ok": False, "error": "text is required"}
    return tenant_manager.store(
        t.tenant_id, text=text,
        source=req.get("source", "tenant"),
        tags=req.get("tags", []),
    )


@app.get("/api/tenants/me/recall")
async def tenant_recall(
    q: str,
    limit: int = 20,
    include_channels: bool = True,
    token: Optional[str] = Depends(_extract_token),
):
    """Semantic recall across the caller's own memory + subscribed channels."""
    try:
        t = tenant_manager.require_role(token, "admin", "user", "readonly")
    except PermissionError as e:
        return {"ok": False, "error": str(e)}
    results = tenant_manager.recall(
        t.tenant_id, query=q, limit=limit, include_channels=include_channels,
    )
    return {
        "tenant": t.public_dict(),
        "results": results,
        "count": len(results),
    }


@app.post("/api/channels")
async def channel_create(req: dict):
    name = (req.get("name") or "").strip()
    if not name:
        return {"ok": False, "error": "name is required"}
    ch = tenant_manager.create_channel(name=name, description=req.get("description", ""))
    return {"ok": True, "channel": ch.to_dict()}


@app.get("/api/channels")
async def channel_list():
    return {"channels": [c.to_dict() for c in tenant_manager.list_channels()]}


@app.delete("/api/channels/{channel_id}")
async def channel_delete(channel_id: str):
    return {"ok": tenant_manager.delete_channel(channel_id)}


@app.post("/api/channels/{channel_id}/publish")
async def channel_publish(channel_id: str, req: dict,
                          token: Optional[str] = Depends(_extract_token)):
    """Publish a memory into a channel. All subscribed tenants will see it."""
    try:
        t = tenant_manager.require_role(token, "admin", "user")
    except PermissionError as e:
        return {"ok": False, "error": str(e)}
    text = (req.get("text") or "").strip()
    if not text:
        return {"ok": False, "error": "text is required"}
    return tenant_manager.publish_to_channel(
        channel_id, text=text,
        source=f"tenant:{t.name}",
        tags=req.get("tags", []),
    )


@app.post("/api/channels/{channel_id}/subscribe")
async def channel_subscribe(channel_id: str,
                            token: Optional[str] = Depends(_extract_token)):
    try:
        t = tenant_manager.require_role(token, "admin", "user", "readonly")
    except PermissionError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": tenant_manager.subscribe(t.tenant_id, channel_id)}


@app.post("/api/channels/{channel_id}/unsubscribe")
async def channel_unsubscribe(channel_id: str,
                              token: Optional[str] = Depends(_extract_token)):
    try:
        t = tenant_manager.require_role(token, "admin", "user", "readonly")
    except PermissionError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": tenant_manager.unsubscribe(t.tenant_id, channel_id)}


@app.get("/api/tenants/stats")
async def tenant_stats():
    return tenant_manager.stats()


# --- V1.3 Security endpoints ---

@app.get("/api/audit/tail")
async def audit_tail(limit: int = 100):
    """Return the last N audit entries (admin-only in production)."""
    return {"entries": audit_log.tail(limit)}


@app.get("/api/audit/verify")
async def audit_verify():
    """Walk the full audit chain, verify tamper-evident integrity."""
    return audit_log.verify_chain()


@app.get("/api/security/rate-limit/{key}")
async def rate_limit_status(key: str):
    """How many tokens remain in the bucket for this key."""
    return {"key": key, "tokens_remaining": rate_limiter.remaining(key)}


# --- V1.3 Benchmarks ---

@app.get("/api/benchmarks")
async def benchmarks_latest():
    """Return the most recent benchmark results. None if never run."""
    latest = bench_mod.load_results()
    if latest is None:
        return {"ok": False, "error": "No benchmark results yet. POST /api/benchmarks/run to generate."}
    return {"ok": True, "results": latest}


@app.post("/api/benchmarks/run")
async def benchmarks_run():
    """Run the full benchmark suite synchronously and persist the results.
    Takes 5-10 seconds."""
    r = bench_mod.run_all()
    path = bench_mod.save_results(r)
    return {
        "ok": True,
        "saved_to": path,
        "results": r.to_dict(),
        "markdown": bench_mod.render_markdown(r),
    }


# --- V1.3 Sleep consolidation endpoints ---

@app.get("/api/sleep/status")
async def sleep_status():
    """Status of the autonomous-thinking background thread."""
    if sleep_consolidator is None:
        return {"ok": False, "error": "sleep consolidator not running"}
    return sleep_consolidator.status()


@app.get("/api/sleep/insights")
async def sleep_insights(limit: int = 20):
    """Recent autonomously-generated insights (cross-refs + patterns)."""
    if sleep_consolidator is None:
        return {"insights": []}
    return {"insights": sleep_consolidator.recent_insights(limit=limit)}


@app.post("/api/sleep/consolidate")
async def sleep_force_consolidation():
    """Force an immediate consolidation pass (normally runs on idle)."""
    if sleep_consolidator is None:
        return {"ok": False, "error": "sleep consolidator not running"}
    r = sleep_consolidator.consolidate()
    return {"ok": True, "pass": r.to_dict()}


@app.get("/api/sleep/history")
async def sleep_history(limit: int = 20):
    if sleep_consolidator is None:
        return {"history": []}
    return {"history": sleep_consolidator.history(limit=limit)}


# --- V1.3 Webhooks + Event Stream ---

@app.post("/api/webhooks")
async def webhook_register(req: dict):
    url = (req.get("url") or "").strip()
    if not url or not url.startswith(("http://", "https://")):
        return {"ok": False, "error": "valid url is required"}
    event_types = req.get("event_types") or ["*"]
    secret = req.get("secret", "")
    wh = webhook_manager.register(url=url, event_types=event_types, secret=secret)
    return {"ok": True, "webhook": wh.to_dict()}


@app.get("/api/webhooks")
async def webhooks_list():
    return {"webhooks": [w.to_dict() for w in webhook_manager.list_webhooks()]}


@app.delete("/api/webhooks/{webhook_id}")
async def webhook_unregister(webhook_id: str):
    return {"ok": webhook_manager.unregister(webhook_id)}


@app.get("/api/events/recent")
async def events_recent(limit: int = 50, type: Optional[str] = None):
    """Last N events from the in-memory bus (for debugging + sync clients)."""
    return {"events": event_bus.recent(limit=limit, event_type=type)}


@app.get("/api/events/stream")
async def events_stream():
    """Server-Sent Events stream. Clients hold this open and get live events."""
    from fastapi.responses import StreamingResponse
    import asyncio

    event_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _on_event(ev):
        try:
            loop.call_soon_threadsafe(event_queue.put_nowait, ev)
        except Exception:
            pass

    unsub = event_bus.subscribe(_on_event)

    async def _gen():
        try:
            # Initial comment to open stream
            yield ": connected\n\n"
            while True:
                try:
                    ev = await asyncio.wait_for(event_queue.get(), timeout=30.0)
                    data = json.dumps(ev.to_dict())
                    yield f"event: {ev.type}\ndata: {data}\n\n"
                except asyncio.TimeoutError:
                    # Heartbeat comment so proxies don't close the connection
                    yield ": keepalive\n\n"
        finally:
            unsub()

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.get("/api/plugins")
async def plugins_list():
    """List all discovered/loaded plugins and the tools they added."""
    from dataclasses import asdict
    return {"plugins": [asdict(p) for p in list_plugins()]}


@app.get("/api/benchmarks/markdown")
async def benchmarks_markdown():
    """Return the latest results as Markdown (Discord-ready)."""
    latest = bench_mod.load_results()
    if latest is None:
        return JSONResponse({"ok": False, "error": "No results yet"})
    # Build a throwaway BenchmarkResults so render_markdown can format it
    r = bench_mod.BenchmarkResults(**{
        k: latest.get(k, {}) for k in
        ["memory_recall", "query_latency", "action_success",
         "discrimination", "hardware"]
    })
    r.ran_at = latest.get("ran_at", 0)
    r.duration_sec = latest.get("duration_sec", 0)
    return {"markdown": bench_mod.render_markdown(r)}


# =============================================================================
# V1.3 — Full-Duplex Voice Pipeline
# =============================================================================

def _ensure_voice_pipeline():
    """Lazy-init so the voice backends (possibly heavy ML libs) only load when
    someone actually uses the voice API."""
    global voice_pipeline
    if voice_pipeline is not None:
        return voice_pipeline

    def _handle_utterance(user_text: str) -> str:
        # Default handler: observe via Claude bridge and return the brain's response.
        try:
            if claude_bridge is not None:
                claude_bridge.observe(user_text, source="voice")
            # Simple echo-ack — host apps can override by setting a custom handler.
            return f"Noted: {user_text[:80]}"
        except Exception as e:
            return f"(voice handler error: {e})"

    voice_pipeline = VoicePipeline(on_utterance=_handle_utterance)
    print(
        f"[VOICE] Pipeline ready — "
        f"VAD={voice_pipeline.vad.backend_name} "
        f"STT={voice_pipeline.stt.backend_name} "
        f"TTS={voice_pipeline.tts.backend_name}"
    )
    return voice_pipeline


@app.post("/api/voice/start")
async def voice_start():
    """Start the voice pipeline (LISTENING state)."""
    pipe = _ensure_voice_pipeline()
    pipe.start()
    return {"ok": True, "state": pipe.state.value}


@app.post("/api/voice/stop")
async def voice_stop():
    """Stop the voice pipeline."""
    if voice_pipeline is not None:
        voice_pipeline.stop()
    return {"ok": True}


@app.get("/api/voice/status")
async def voice_status():
    """Full pipeline status incl. backends, state, and stats."""
    pipe = _ensure_voice_pipeline()
    return pipe.status()


@app.post("/api/voice/speak")
async def voice_speak(req: dict):
    """Speak a string — TTS via the configured backend."""
    pipe = _ensure_voice_pipeline()
    text = req.get("text", "").strip()
    if not text:
        return {"ok": False, "error": "text is required"}
    try:
        result = pipe.speak(text)
        return {
            "ok": True,
            "backend": result.backend,
            "sample_rate": result.sample_rate,
            "duration_sec": round(result.duration_sec, 3),
            "wav_bytes": len(result.wav_bytes),
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@app.post("/api/voice/feed")
async def voice_feed(req: dict):
    """
    Feed raw PCM16 audio into the pipeline.
    req = {"pcm_base64": "<base64-encoded 16-bit mono 16kHz>"}
    Returns {"event": <utterance or null>, "state": ...}.
    """
    pipe = _ensure_voice_pipeline()
    import base64
    pcm_b64 = req.get("pcm_base64", "")
    if not pcm_b64:
        return {"ok": False, "error": "pcm_base64 is required"}
    try:
        pcm = base64.b64decode(pcm_b64)
    except Exception as e:
        return {"ok": False, "error": f"bad base64: {e}"}
    event = pipe.feed_audio(pcm)
    return {
        "ok": True,
        "state": pipe.state.value,
        "event": None if event is None else {
            "text": event.text,
            "duration_sec": round(event.duration_sec, 2),
            "barge_in": event.barge_in,
            "vad_state": event.vad_state,
        },
    }


# =============================================================================
# WebSocket for real-time streaming
# =============================================================================

# FIXED server.py (WebSocket section only)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.add(ws)
    print(f"[WS] Client connected ({len(connected_clients)} total)")

    try:
        positions = brain.get_neuron_positions()
        await ws.send_json({"type": "init", "positions": positions})
    except Exception:
        pass

    try:
        update_interval = 1.0 / BrainConfig.WS_UPDATE_RATE
        while True:
            start = time.time()

            state = brain.get_state()

            if claude_bridge:
                state["claude"] = {
                    "connected": True,
                    "interactions": claude_bridge._interaction_count,
                }

            await ws.send_json({"type": "state", "data": state})

            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=0.001)

                if msg.get("type") == "text_input":
                    text = msg.get("text", "")
                    print(f"[INPUT] {text}")

                    # 🧠 inject into brain
                    features = text_encoder.encode(text)
                    brain.inject_sensory_input("text", features)

                    if claude_bridge:
                        claude_bridge.send_observation({
                            "type": "text",
                            "content": text,
                            "source": "dashboard",
                        })

                    try:
                        result = ask_ollama(text)
                        print("OLLAMA RESULT:", result)

                        if not result:
                            result = "No response from AI"

                    except Exception as e:
                        result = f"Error: {str(e)}"

                    # 🔥 FINAL FIX — SEND RESPONSE
                    await ws.send_json({
                        "type": "response",
                        "data": result
                    })
                elif msg.get("type") == "command":
                    cmd = msg.get("cmd")

                    if cmd == "start_vision":
                        vision_encoder.start_webcam()
                    elif cmd == "stop_vision":
                        vision_encoder.stop_webcam()
                    elif cmd == "start_audio":
                        audio_encoder.start_microphone()
                    elif cmd == "stop_audio":
                        audio_encoder.stop_microphone()
                    elif cmd == "start_screen":
                        screen_observer.start()
                    elif cmd == "stop_screen":
                        screen_observer.stop()
                    elif cmd == "start_video":
                        if video_recorder:
                            video_recorder.start()
                    elif cmd == "stop_video":
                        if video_recorder:
                            video_recorder.stop()
                    elif cmd == "save":
                        save_brain(brain)
                    elif cmd == "load":
                        load_brain(brain)

            except asyncio.TimeoutError:
                pass

            elapsed = time.time() - start
            sleep_time = update_interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except WebSocketDisconnect:
        pass
    finally:
        connected_clients.discard(ws)
