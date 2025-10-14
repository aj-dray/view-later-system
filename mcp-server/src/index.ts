#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from "@modelcontextprotocol/sdk/types.js";

// Backend API configuration
const BACKEND_URL = process.env.LATER_BACKEND_URL || "http://localhost:8000";
const ACCESS_TOKEN = process.env.LATER_ACCESS_TOKEN;

if (!ACCESS_TOKEN) {
  console.error("Error: LATER_ACCESS_TOKEN environment variable is required");
  process.exit(1);
}

// API client helper
async function apiRequest(
  endpoint: string,
  options: RequestInit = {}
): Promise<any> {
  const url = `${BACKEND_URL}${endpoint}`;
  const headers = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${ACCESS_TOKEN}`,
    ...options.headers,
  };

  const response = await fetch(url, { ...options, headers });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `API request failed: ${response.status} ${response.statusText}\n${errorText}`
    );
  }

  return response.json();
}

// Create MCP server
const server = new Server(
  {
    name: "later-system",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Define available tools
const TOOLS: Tool[] = [
  {
    name: "get_queue",
    description:
      "Retrieve items from your queue with flexible filtering and ordering. " +
      "Use this to get queued items, filter by status, or order by priority/date. " +
      "Returns items with title, url, and summary.",
    inputSchema: {
      type: "object",
      properties: {
        status: {
          type: "string",
          description:
            "Filter by client status. Options: 'adding', 'queued', 'paused', 'completed', 'bookmark', 'error'",
          enum: ["adding", "queued", "paused", "completed", "bookmark", "error"],
        },
        order_by: {
          type: "string",
          description: "Column to order by (e.g., 'created_at', 'expiry_score')",
          default: "created_at",
        },
        order: {
          type: "string",
          description: "Sort direction: 'asc' or 'desc'",
          enum: ["asc", "desc"],
          default: "desc",
        },
        limit: {
          type: "number",
          description: "Maximum number of items to return (1-200)",
          minimum: 1,
          maximum: 200,
          default: 50,
        },
        offset: {
          type: "number",
          description: "Number of items to skip (for pagination)",
          minimum: 0,
          default: 0,
        },
      },
    },
  },
  {
    name: "semantic_search",
    description:
      "Search items using semantic/vector similarity. " +
      "Finds content that is conceptually similar to your query, even if exact keywords don't match. " +
      "Returns relevant items with title, url, and summary.",
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Search query - describe what you're looking for",
        },
        limit: {
          type: "number",
          description: "Maximum number of results to return (1-10)",
          minimum: 1,
          maximum: 10,
          default: 10,
        },
      },
      required: ["query"],
    },
  },
  {
    name: "add_item",
    description:
      "Add a new item to the queue by URL. " +
      "The system will automatically extract content, generate a summary, and create embeddings. " +
      "Returns the created item's ID.",
    inputSchema: {
      type: "object",
      properties: {
        url: {
          type: "string",
          description: "URL of the content to add",
        },
      },
      required: ["url"],
    },
  },
  {
    name: "update_item_status",
    description:
      "Update the status of one or more items. " +
      "Use this to mark items as completed, pause them, bookmark them, etc. " +
      "Returns update results for each item.",
    inputSchema: {
      type: "object",
      properties: {
        item_ids: {
          type: "array",
          items: {
            type: "string",
          },
          description: "Array of item IDs to update",
        },
        status: {
          type: "string",
          description: "New status to set",
          enum: ["adding", "queued", "paused", "completed", "bookmark", "error"],
        },
      },
      required: ["item_ids", "status"],
    },
  },
];

// Tool handlers
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools: TOOLS };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case "get_queue": {
        const {
          status,
          order_by = "created_at",
          order = "desc",
          limit = 50,
          offset = 0,
        } = args as any;

        // Build filter query parameters
        const filters: string[] = [];
        if (status) {
          filters.push(`client_status:=:${status}`);
        }

        const queryParams = new URLSearchParams({
          order_by,
          order,
          limit: String(limit),
          offset: String(offset),
        });

        if (filters.length > 0) {
          filters.forEach((f) => queryParams.append("filter", f));
        }

        // Always request title, url, summary
        ["title", "url", "summary", "id", "client_status", "created_at"].forEach(
          (col) => queryParams.append("columns", col)
        );

        const items = await apiRequest(`/items/select?${queryParams.toString()}`);

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(items, null, 2),
            },
          ],
        };
      }

      case "semantic_search": {
        const { query, limit = 10 } = args as any;

        if (!query) {
          throw new Error("Query parameter is required");
        }

        const queryParams = new URLSearchParams({
          query,
          mode: "semantic",
          limit: String(limit),
        });

        // Request title, url, summary columns
        ["title", "url", "summary"].forEach((col) =>
          queryParams.append("columns", col)
        );

        const result = await apiRequest(`/items/search?${queryParams.toString()}`);

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(result.results, null, 2),
            },
          ],
        };
      }

      case "add_item": {
        const { url } = args as any;

        if (!url) {
          throw new Error("URL parameter is required");
        }

        const result = await apiRequest("/items/add", {
          method: "POST",
          body: JSON.stringify({ url }),
        });

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(result, null, 2),
            },
          ],
        };
      }

      case "update_item_status": {
        const { item_ids, status } = args as any;

        if (!item_ids || !Array.isArray(item_ids) || item_ids.length === 0) {
          throw new Error("item_ids must be a non-empty array");
        }

        if (!status) {
          throw new Error("status parameter is required");
        }

        const result = await apiRequest("/items/update", {
          method: "POST",
          body: JSON.stringify({
            item_ids,
            updates: { client_status: status },
          }),
        });

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(result, null, 2),
            },
          ],
        };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text: `Error: ${error instanceof Error ? error.message : String(error)}`,
        },
      ],
      isError: true,
    };
  }
});

// Start the server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Later System MCP server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
