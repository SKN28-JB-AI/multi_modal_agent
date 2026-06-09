docker build -t multi-modal-agent .
docker run -d --name multi-modal-agent `
  -p 8000:8000 `
  -v "$(pwd)/.env:/app/.env:ro" `
  -v "$(pwd)/data:/app/data" `
  --restart unless-stopped `
  multi-modal-agent