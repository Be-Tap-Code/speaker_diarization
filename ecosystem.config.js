module.exports = {
  apps: [
    {
      name: "diarize-deepgram",           // app.py - Deepgram API version
      script: "python",
      args: "app.py",
      cwd: "/root/speaker_diarization",
      interpreter: "none",
      env: {
        PYTHONUNBUFFERED: "1",            // Ensure Python outputs are not buffered
        PATH: "/root/speaker_diarization/venv/bin:$PATH",  // Use venv
      },
      max_memory_restart: "500M",         // App restarts if exceeds 500MB RAM
      instances: 1,
      autorestart: true,
      error_file: "./logs/app_deepgram_error.log",
      out_file: "./logs/app_deepgram_out.log",
      merge_logs: true,
    },
    {
      name: "diarize-local",              // app_api.py - Local processing version
      script: "python",
      args: "app_api.py",
      cwd: "/root/speaker_diarization",
      interpreter: "none",
      env: {
        PYTHONUNBUFFERED: "1",
        PATH: "/root/speaker_diarization/venv/bin:$PATH",
      },
      max_memory_restart: "15G",          // App restarts if exceeds 15GB RAM
      instances: 1,
      autorestart: true,
      error_file: "./logs/app_api_error.log",
      out_file: "./logs/app_api_out.log",
      merge_logs: true,
    },
  ],
};