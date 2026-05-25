# Mem0 Memory Provider

Dual-backend memory provider for Hermes:

- **Local mode**: `mem0.Memory` + MiniMax + FastEmbed + local ChromaDB
- **Cloud mode**: `mem0.MemoryClient` + Mem0 Platform API

Local mode is useful when you want durable memory without depending on Mem0 cloud.

## Requirements

### Local mode

- `pip install mem0ai chromadb fastembed`
- MiniMax API key in `~/.hermes/.env`:

```bash
echo 'MINIMAX_API_KEY=your-key' >> ~/.hermes/.env
```

### Cloud mode

- `pip install mem0ai`
- Mem0 API key from [app.mem0.ai](https://app.mem0.ai)

```bash
echo 'MEM0_API_KEY=your-key' >> ~/.hermes/.env
```

## Setup

```bash
hermes config set memory.provider mem0
```

## Config

Config file: `$HERMES_HOME/mem0.json`

### Local mode example

```json
{
  "local_mode": true,
  "minimax_base_url": "https://api.minimaxi.com/anthropic",
  "chroma_path": "~/.hermes/mem0_chroma",
  "collection_name": "hermes_memories",
  "user_id": "hermes-user",
  "agent_id": "hermes",
  "rerank": true
}
```

### Cloud mode example

```json
{
  "local_mode": false,
  "user_id": "hermes-user",
  "agent_id": "hermes",
  "rerank": true
}
```

### Keys

- `local_mode`: `true` for MiniMax+Chroma local mode, `false` for Mem0 cloud mode
- `minimax_base_url`: MiniMax API base URL for local mode
- `chroma_path`: local ChromaDB directory
- `collection_name`: Chroma collection name
- `user_id`: memory namespace user id
- `agent_id`: attribution agent id
- `rerank`: enable reranking for recall
- `api_key`: optional Mem0 cloud API key when using cloud mode

## Tools

- `mem0_profile`: list all stored memories for the current user
- `mem0_search`: semantic search over stored memories
- `mem0_conclude`: store a fact verbatim without LLM extraction
