This agent uses a local LLM (via Ollama) to generate BDD test plans from a user story.

Prerequisites
- Docker
- Docker Compose

How to Run:

Configure Model (Optional): The model is set to `phi3:mini` in the `.env` file. Change it if you wish.

Build and Run:
```bash
docker-compose up --build
```
