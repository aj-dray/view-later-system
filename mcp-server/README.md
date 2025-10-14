# Later System MCP Server

A Model Context Protocol (MCP) server that provides Claude with access to your Later System backend. Enables queue management, semantic search, and content operations through natural language.

## Features

The MCP server exposes 4 tools:

### 1. `get_queue`
Retrieve items from your queue with flexible filtering and ordering.

**Parameters:**
- `status` (optional): Filter by status (`adding`, `queued`, `paused`, `completed`, `bookmark`, `error`)
- `order_by` (optional): Column to sort by (default: `created_at`)
- `order` (optional): Sort direction `asc` or `desc` (default: `desc`)
- `limit` (optional): Max items to return, 1-200 (default: 50)
- `offset` (optional): Pagination offset (default: 0)

**Example usage:**
```
"Get my 10 most recent queued items"
"Show me completed items ordered by priority"
```

### 2. `semantic_search`
Search items using vector similarity - finds conceptually similar content.

**Parameters:**
- `query` (required): What you're looking for
- `limit` (optional): Max results, 1-10 (default: 10)

**Example usage:**
```
"Find articles about machine learning"
"Search for content related to database optimization"
```

### 3. `add_item`
Add a new item by URL. The system will automatically extract content, generate summaries, and create embeddings.

**Parameters:**
- `url` (required): URL to add

**Example usage:**
```
"Save this URL to my queue: https://news.ycombinator.com/item?id=12345"
```

### 4. `update_item_status`
Update the status of one or more items.

**Parameters:**
- `item_ids` (required): Array of item IDs
- `status` (required): New status (`adding`, `queued`, `paused`, `completed`, `bookmark`, `error`)

**Example usage:**
```
"Mark items abc123 and def456 as completed"
"Pause item xyz789"
```

## Setup

### 1. Install Dependencies

```bash
cd mcp-server
npm install
```

### 2. Build

```bash
npm run build
```

### 3. Get an Access Token

You need a JWT access token from your Later System backend. You can generate one using:

```bash
# From the backend directory
python tools/generate_access_token.py
```

Or via the API:
```bash
curl -X POST http://localhost:8000/user/access-token \
  -H "Content-Type: application/json" \
  -d '{"username": "your-username", "password": "your-password"}'
```

Save the returned `access_token`.

### 4. Configure Claude Desktop

Add the MCP server to your Claude Desktop config file on MacOS:`~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "later-system": {
      "command": "node",
      "args": [
        "/absolute/path/to/later-system-mcp/mcp-server/dist/index.js"
      ],
      "env": {
        "LATER_ACCESS_TOKEN": "your-jwt-token-here",
        "LATER_BACKEND_URL": "http://localhost:8000"
      }
    }
  }
}
```

### 5. Restart Claude Desktop

After updating the config, restart Claude Desktop to load the MCP server.

## Environment Variables

- `LATER_ACCESS_TOKEN` (required): JWT token for authentication
- `LATER_BACKEND_URL` (optional): Backend URL (default: `http://localhost:8000`)


## Architecture

The MCP server acts as a bridge between Claude and your Later System backend:

```
Claude Desktop → MCP Server → Later System Backend → PostgreSQL
```

- Uses the MCP SDK's stdio transport for local communication
- Authenticates via JWT tokens
- Makes REST API calls to your backend
- Returns formatted responses to Claude
