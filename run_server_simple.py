#!/usr/bin/env python3
"""
Simple server runner without debugger reloader
This avoids the reloader hanging issue
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the Flask app
from nexus_app import app

if __name__ == '__main__':
    import socket
    
    # Find available port
    port = 8080
    for attempt in range(10):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('127.0.0.1', port))
            sock.close()
            break
        except OSError:
            port += 1
    
    print("=" * 60)
    print("🌟 BIT Tutor Educational Platform")
    print("=" * 60)
    print(f"✅ Server starting on port {port}")
    print(f"🔗 Access at: http://127.0.0.1:{port}")
    print(f"📝 Press CTRL+C to stop")
    print("=" * 60)
    
    # Run without debugger and reloader
    app.run(
        debug=False,
        host='127.0.0.1',
        port=port,
        use_reloader=False,
        threaded=True
    )

