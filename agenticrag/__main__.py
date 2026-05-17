"""
agenticrag.__main__ — Entry point for `python -m agenticrag`.

Usage:
    python -m agenticrag serve              # Start the web UI
    python -m agenticrag serve --port 9000  # Custom port
    python -m agenticrag --help             # Show help
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="agenticrag",
        description="AgenticRAG — Vectorless, Reasoning-based RAG",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── serve ─────────────────────────────────────────────────────────────
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the AgenticRAG web UI",
    )
    serve_parser.add_argument(
        "--host", default="0.0.0.0",
        help="Bind address (default: 0.0.0.0 for LAN access)",
    )
    serve_parser.add_argument(
        "--port", type=int, default=8000,
        help="Port (default: 8000)",
    )

    # ── version ───────────────────────────────────────────────────────────
    subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.command == "serve":
        _run_serve(args.host, args.port)
    elif args.command == "version":
        from agenticrag import __version__
        print(f"agenticrag {__version__}")
    else:
        parser.print_help()
        sys.exit(0)


def _run_serve(host: str, port: int):
    """Start the web server."""
    try:
        import uvicorn
    except ImportError:
        print(
            "Error: The web UI requires extra dependencies.\n"
            "Install them with:\n\n"
            "    pip install agenticrag[web]\n"
        )
        sys.exit(1)

    # Import the server app
    import importlib.util
    from pathlib import Path

    # Try to find server.py relative to the package
    server_path = Path(__file__).parent.parent / "server.py"
    if not server_path.exists():
        # Fallback: try current working directory
        server_path = Path.cwd() / "server.py"

    if server_path.exists():
        spec = importlib.util.spec_from_file_location("server", server_path)
        server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(server)
        app = server.app
        server._ensure_dirs()
    else:
        print("Error: server.py not found. Run from the project directory.")
        sys.exit(1)

    import socket

    def _get_lan_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "<your-ip>"

    lan_ip = _get_lan_ip()

    # Print banner BEFORE starting uvicorn (uvicorn.run is blocking)
    print("\n" + "=" * 60)
    print("  AgenticRAG Web UI")
    print("=" * 60)
    print(f"  Local:   http://localhost:{port}")
    print(f"  LAN:     http://{lan_ip}:{port}")
    print("  " + "─" * 54)
    print(f"  Share the LAN URL with devices on your network.\n")

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
