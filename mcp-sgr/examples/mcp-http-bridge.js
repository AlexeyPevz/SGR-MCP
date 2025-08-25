#!/usr/bin/env node
/**
 * MCP HTTP Bridge for MCP-SGR
 * 
 * This bridge allows Claude Desktop to connect to MCP-SGR running on a remote VPS
 * via HTTP API instead of stdio.
 */

const readline = require('readline');
const https = require('https');
const http = require('http');

// Configuration from environment
const SGR_HTTP_URL = process.env.SGR_HTTP_URL || 'http://localhost:8080';
const SGR_API_KEY = process.env.SGR_API_KEY || '';

// Parse URL
const url = new URL(SGR_HTTP_URL);
const client = url.protocol === 'https:' ? https : http;

// Setup stdio interface
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false
});

// Helper to send JSON-RPC response
function sendResponse(id, result, error = null) {
  const response = {
    jsonrpc: '2.0',
    id: id
  };
  
  if (error) {
    response.error = error;
  } else {
    response.result = result;
  }
  
  console.log(JSON.stringify(response));
}

// Map MCP methods to HTTP endpoints
const methodMap = {
  'initialize': async () => ({
    protocolVersion: '1.0',
    serverInfo: {
      name: 'sgr-reasoning-http',
      version: '1.0.0'
    }
  }),
  
  'list_tools': async () => {
    // Return the tool definitions from mcp.json
    return {
      tools: [
        {
          name: 'apply_sgr',
          description: 'Apply a structured reasoning schema to analyze and structure a task',
          inputSchema: {
            type: 'object',
            properties: {
              task: { type: 'string', description: 'The task or problem to analyze' },
              context: { type: 'object', description: 'Additional context information' },
              schema_type: { 
                type: 'string', 
                enum: ['auto', 'analysis', 'planning', 'decision', 'search', 'code_generation', 'summarization', 'custom'],
                default: 'auto'
              },
              budget: { type: 'string', enum: ['none', 'lite', 'full'], default: 'lite' }
            },
            required: ['task']
          }
        },
        {
          name: 'enhance_prompt_with_sgr',
          description: 'Transform a simple prompt into a structured prompt with SGR guidance',
          inputSchema: {
            type: 'object',
            properties: {
              original_prompt: { type: 'string', description: 'The original simple prompt' },
              target_model: { type: 'string', description: 'Target model identifier' }
            },
            required: ['original_prompt']
          }
        }
      ]
    };
  },
  
  'call_tool': async (params) => {
    const { name, arguments: args } = params;
    
    // Map tool calls to HTTP endpoints
    let endpoint, body;
    
    switch (name) {
      case 'apply_sgr':
        endpoint = '/v1/apply-sgr';
        body = args;
        break;
        
      case 'enhance_prompt_with_sgr':
        endpoint = '/v1/enhance-prompt';
        body = args;
        break;
        
      default:
        throw new Error(`Unknown tool: ${name}`);
    }
    
    // Make HTTP request
    return new Promise((resolve, reject) => {
      const options = {
        hostname: url.hostname,
        port: url.port,
        path: endpoint,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(SGR_API_KEY && { 'X-API-Key': SGR_API_KEY })
        }
      };
      
      const req = client.request(options, (res) => {
        let data = '';
        
        res.on('data', (chunk) => {
          data += chunk;
        });
        
        res.on('end', () => {
          try {
            const result = JSON.parse(data);
            if (res.statusCode === 200) {
              resolve(result);
            } else {
              reject(new Error(result.detail || 'HTTP error'));
            }
          } catch (e) {
            reject(new Error('Invalid JSON response'));
          }
        });
      });
      
      req.on('error', reject);
      req.write(JSON.stringify(body));
      req.end();
    });
  }
};

// Process incoming JSON-RPC requests
rl.on('line', async (line) => {
  try {
    const request = JSON.parse(line);
    const { id, method, params } = request;
    
    if (methodMap[method]) {
      try {
        const result = await methodMap[method](params);
        sendResponse(id, result);
      } catch (error) {
        sendResponse(id, null, {
          code: -32603,
          message: error.message
        });
      }
    } else {
      sendResponse(id, null, {
        code: -32601,
        message: `Method not found: ${method}`
      });
    }
  } catch (error) {
    // Invalid JSON
    sendResponse(null, null, {
      code: -32700,
      message: 'Parse error'
    });
  }
});

// Handle graceful shutdown
process.on('SIGINT', () => {
  process.exit(0);
});

// Log startup
process.stderr.write(`MCP-SGR HTTP Bridge started. Connecting to: ${SGR_HTTP_URL}\n`);