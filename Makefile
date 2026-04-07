# Jupyter + MCP Infrastructure Management
# Usage:
#   make start    - Start Jupyter server
#   make stop     - Stop Jupyter server
#   make status   - Check server status
#   make restart  - Restart server
#   make logs     - View server logs

# Configuration
JUPYTER_PORT := 8888
JUPYTER_TOKEN := claude-jupyter-token
PID_FILE := /tmp/jupyter-server.pid
LOG_FILE := /tmp/jupyter-server.log
SETTINGS_FILE := $(HOME)/.claude/settings.json

# Auto-detect Python environment: uv > .venv > system
HAS_UV   := $(shell command -v uv >/dev/null 2>&1 && echo 1 || echo 0)
HAS_VENV := $(shell [ -d .venv ] && echo 1 || echo 0)

ifeq ($(HAS_UV),1)
  PYTHON  := uv run python
  ENV_MSG := uv managed environment
else ifeq ($(HAS_VENV),1)
  PYTHON  := .venv/bin/python
  ENV_MSG := local venv (.venv)
else
  PYTHON  := python3
  ENV_MSG := system Python
endif

.PHONY: start stop status restart logs help

# Default target
help:
	@echo "Jupyter + MCP Infrastructure Management"
	@echo ""
	@echo "Available commands:"
	@echo "  make start    - Start Jupyter server"
	@echo "  make stop     - Stop Jupyter server"
	@echo "  make status   - Check server status"
	@echo "  make restart  - Restart server (stop + start)"
	@echo "  make logs     - View server logs"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Detected environment: $(ENV_MSG)"

# Start Jupyter server
start:
	@echo "🚀 Starting Jupyter + MCP Infrastructure..."
	@echo ""
	@echo "1️⃣  Verifying dependencies..."
	@$(PYTHON) -c "import jupyter_server" 2>/dev/null || { \
		echo "❌ Error: jupyter-server is not installed"; \
		echo "Run: uv add jupyter-server  (or: pip install jupyter-server)"; \
		exit 1; \
	}
	@$(PYTHON) -c "import mcp_jupyter" 2>/dev/null || { \
		echo "❌ Error: mcp-jupyter is not installed"; \
		echo "Run: uv add mcp-jupyter  (or: pip install mcp-jupyter)"; \
		exit 1; \
	}
	@test -f "$(SETTINGS_FILE)" || { \
		echo "❌ Error: Claude settings file not found at $(SETTINGS_FILE)"; \
		exit 1; \
	}
	@grep -q "jupyter-local" "$(SETTINGS_FILE)" || { \
		echo "❌ Error: jupyter-local MCP server not configured in $(SETTINGS_FILE)"; \
		exit 1; \
	}
	@echo "✅ All dependencies verified ($(ENV_MSG))"
	@echo ""
	@echo "2️⃣  Starting Jupyter server..."
	@if [ -f "$(PID_FILE)" ]; then \
		PID=$$(cat $(PID_FILE)); \
		if ps -p $$PID > /dev/null 2>&1; then \
			echo "⚠️  Jupyter server is already running (PID: $$PID)"; \
			echo "📍 Server URL: http://localhost:$(JUPYTER_PORT)/?token=$(JUPYTER_TOKEN)"; \
			echo ""; \
			echo "To view logs: tail -f $(LOG_FILE)"; \
			exit 0; \
		else \
			echo "🧹 Cleaning up stale PID file..."; \
			rm "$(PID_FILE)"; \
		fi; \
	fi
	@if lsof -Pi :$(JUPYTER_PORT) -sTCP:LISTEN -t >/dev/null 2>&1; then \
		echo "❌ Error: Port $(JUPYTER_PORT) is already in use"; \
		echo "Run: lsof -i :$(JUPYTER_PORT) to see what's using it"; \
		exit 1; \
	fi
	@echo "🔧 Configuring kernel..."
	@$(PYTHON) -m ipykernel install --user --name=kaggle-venv --display-name="Python ($(ENV_MSG))" 2>/dev/null || true
	@echo "✅ Kernel registered"
	@echo "📝 Starting Jupyter server on port $(JUPYTER_PORT)..."
	@echo "🐍 Using $(ENV_MSG)"
	@nohup $(PYTHON) -m jupyter_server \
		--no-browser \
		--port=$(JUPYTER_PORT) \
		--ServerApp.token="$(JUPYTER_TOKEN)" \
		--ServerApp.allow_origin='*' \
		--ServerApp.disable_check_xsrf=False \
		> "$(LOG_FILE)" 2>&1 & \
	echo $$! > "$(PID_FILE)"
	@echo "⏳ Waiting for server to start..."
	@sleep 3
	@if ! ps -p $$(cat $(PID_FILE)) > /dev/null 2>&1; then \
		echo "❌ Failed to start Jupyter server"; \
		echo "Check logs: cat $(LOG_FILE)"; \
		rm "$(PID_FILE)"; \
		exit 1; \
	fi
	@echo "✅ Jupyter server started (PID: $$(cat $(PID_FILE)))"
	@echo ""
	@echo "3️⃣  Verifying Jupyter API..."
	@if curl -s -f "http://localhost:$(JUPYTER_PORT)/api?token=$(JUPYTER_TOKEN)" > /dev/null 2>&1; then \
		API_VERSION=$$(curl -s "http://localhost:$(JUPYTER_PORT)/api?token=$(JUPYTER_TOKEN)" | $(PYTHON) -c "import sys, json; print(json.load(sys.stdin)['version'])" 2>/dev/null || echo "unknown"); \
		echo "✅ Jupyter API responding (v$$API_VERSION)"; \
	else \
		echo "⚠️  Warning: Jupyter API not responding yet (may need more time)"; \
	fi
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "✨ Jupyter + MCP Infrastructure is ready!"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "📍 Jupyter Server:"
	@echo "   URL:   http://localhost:$(JUPYTER_PORT)/?token=$(JUPYTER_TOKEN)"
	@echo "   Token: $(JUPYTER_TOKEN)"
	@echo "   PID:   $$(cat $(PID_FILE))"
	@echo "   Logs:  $(LOG_FILE)"
	@echo ""
	@echo "🛑 To stop: make stop"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Stop Jupyter server
stop:
	@echo "🛑 Stopping Jupyter + MCP Infrastructure..."
	@echo ""
	@if [ ! -f "$(PID_FILE)" ]; then \
		echo "⚠️  No PID file found at $(PID_FILE)"; \
		if lsof -Pi :$(JUPYTER_PORT) -sTCP:LISTEN -t >/dev/null 2>&1; then \
			echo "🔍 Found Jupyter server running on port $(JUPYTER_PORT)"; \
			PID=$$(lsof -Pi :$(JUPYTER_PORT) -sTCP:LISTEN -t); \
			echo "📋 PID: $$PID"; \
			kill $$PID; \
			echo "✅ Jupyter server stopped"; \
		else \
			echo "ℹ️  No Jupyter server appears to be running on port $(JUPYTER_PORT)"; \
		fi; \
		exit 0; \
	fi
	@PID=$$(cat $(PID_FILE)); \
	if ps -p $$PID > /dev/null 2>&1; then \
		echo "📋 Found Jupyter server (PID: $$PID)"; \
		echo "🔪 Sending SIGTERM..."; \
		kill $$PID; \
		sleep 2; \
		if ps -p $$PID > /dev/null 2>&1; then \
			echo "⚠️  Process still running, sending SIGKILL..."; \
			kill -9 $$PID; \
			sleep 1; \
		fi; \
		if ps -p $$PID > /dev/null 2>&1; then \
			echo "❌ Failed to stop Jupyter server"; \
			exit 1; \
		else \
			echo "✅ Jupyter server stopped successfully"; \
		fi; \
	else \
		echo "ℹ️  Jupyter server (PID: $$PID) is not running"; \
	fi
	@if [ -f "$(PID_FILE)" ]; then \
		rm "$(PID_FILE)"; \
		echo "🧹 Cleaned up PID file"; \
	fi
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "✅ Jupyter + MCP Infrastructure stopped"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "🚀 To restart: make start"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check server status
status:
	@echo "🔍 Checking Jupyter + MCP Infrastructure Status..."
	@echo ""
	@echo "1️⃣  Environment: $(ENV_MSG)"
	@echo ""
	@echo "2️⃣  Dependencies:"
	@if $(PYTHON) -c "import jupyter_server; print(jupyter_server.__version__)" 2>/dev/null; then \
		JUPYTER_VERSION=$$($(PYTHON) -c "import jupyter_server; print(jupyter_server.__version__)" 2>/dev/null || echo "unknown"); \
		echo "   ✅ jupyter-server: installed (v$$JUPYTER_VERSION)"; \
	else \
		echo "   ❌ jupyter-server: not installed (run: uv add jupyter-server)"; \
	fi
	@if $(PYTHON) -c "import mcp_jupyter" 2>/dev/null; then \
		MCP_VERSION=$$($(PYTHON) -c "import mcp_jupyter; print(getattr(mcp_jupyter, '__version__', 'unknown'))" 2>/dev/null || echo "unknown"); \
		echo "   ✅ mcp-jupyter: installed (v$$MCP_VERSION)"; \
	else \
		echo "   ❌ mcp-jupyter: not installed"; \
	fi
	@echo ""
	@echo "3️⃣  Jupyter Server:"
	@if [ -f "$(PID_FILE)" ]; then \
		PID=$$(cat $(PID_FILE)); \
		if ps -p $$PID > /dev/null 2>&1; then \
			echo "   ✅ Running (PID: $$PID)"; \
			echo "   📍 URL: http://localhost:$(JUPYTER_PORT)/?token=$(JUPYTER_TOKEN)"; \
			if curl -s -f "http://localhost:$(JUPYTER_PORT)/api?token=$(JUPYTER_TOKEN)" > /dev/null 2>&1; then \
				API_VERSION=$$(curl -s "http://localhost:$(JUPYTER_PORT)/api?token=$(JUPYTER_TOKEN)" | $(PYTHON) -c "import sys, json; print(json.load(sys.stdin)['version'])" 2>/dev/null || echo "unknown"); \
				echo "   ✅ API responding (v$$API_VERSION)"; \
			else \
				echo "   ⚠️  API not responding"; \
			fi; \
		else \
			echo "   ❌ Not running (stale PID file)"; \
		fi; \
	else \
		if lsof -Pi :$(JUPYTER_PORT) -sTCP:LISTEN -t >/dev/null 2>&1; then \
			SERVER_PID=$$(lsof -Pi :$(JUPYTER_PORT) -sTCP:LISTEN -t); \
			echo "   ⚠️  Running (PID: $$SERVER_PID) but no PID file"; \
		else \
			echo "   ❌ Not running"; \
		fi; \
	fi
	@echo ""
	@echo "4️⃣  MCP Configuration:"
	@if [ -f "$(SETTINGS_FILE)" ]; then \
		echo "   ✅ Settings file exists"; \
		if grep -q "jupyter-local" "$(SETTINGS_FILE)"; then \
			echo "   ✅ jupyter-local server configured"; \
			JUPYTER_URL=$$(grep -A 10 "jupyter-local" "$(SETTINGS_FILE)" | grep "JUPYTER_URL" | cut -d'"' -f4); \
			if [ ! -z "$$JUPYTER_URL" ]; then \
				echo "   📍 Configured URL: $$JUPYTER_URL"; \
			fi; \
		else \
			echo "   ❌ jupyter-local server not configured"; \
		fi; \
	else \
		echo "   ❌ Settings file not found"; \
	fi
	@echo ""
	@echo "5️⃣  Logs:"
	@if [ -f "$(LOG_FILE)" ]; then \
		LOG_SIZE=$$(du -h "$(LOG_FILE)" | cut -f1); \
		echo "   📄 Log file: $(LOG_FILE) ($$LOG_SIZE)"; \
		echo "   View: make logs"; \
	else \
		echo "   ℹ️  No log file (server not started yet)"; \
	fi
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "📋 Quick commands:"
	@echo "   make start   - Start the server"
	@echo "   make stop    - Stop the server"
	@echo "   make restart - Restart the server"
	@echo "   make logs    - View logs"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Restart server
restart: stop
	@sleep 1
	@$(MAKE) start

# View logs
logs:
	@if [ -f "$(LOG_FILE)" ]; then \
		tail -f "$(LOG_FILE)"; \
	else \
		echo "❌ No log file found at $(LOG_FILE)"; \
		echo "Server hasn't been started yet."; \
		exit 1; \
	fi
