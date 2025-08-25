# n8n Integration for MCP-SGR

This directory contains integration guides and examples for using MCP-SGR with n8n workflow automation.

## Quick Start (HTTP API)

MCP-SGR exposes a REST API that can be easily used with n8n's HTTP Request node. No custom node installation required!

## Available Endpoints

### 1. Apply SGR Analysis
- **Method**: POST
- **URL**: `http://your-server:8080/v1/apply-sgr`
- **Body**:
```json
{
  "task": "Your task description",
  "schema_type": "analysis",
  "budget": "lite",
  "context": {}
}
```

### 2. Enhance Prompt
- **Method**: POST  
- **URL**: `http://your-server:8080/v1/enhance-prompt`
- **Body**:
```json
{
  "original_prompt": "Your basic prompt",
  "target_model": "gpt-4"
}
```

### 3. Wrap Agent Call
- **Method**: POST
- **URL**: `http://your-server:8080/v1/wrap-agent`
- **Body**:
```json
{
  "agent_endpoint": "http://your-agent/api",
  "agent_request": {},
  "sgr_config": {
    "schema_type": "auto",
    "pre_analysis": true,
    "post_analysis": true
  }
}
```

## Example Workflows

### Basic Analysis Workflow

1. **Manual Trigger** - Start workflow
2. **Set** - Prepare task data
3. **HTTP Request** - Call MCP-SGR API
4. **IF** - Check confidence > 0.7
5. **Set** - Process high confidence results
6. **HTTP Request** - Call enhance prompt for low confidence

### AI Enhancement Pipeline

1. **Webhook** - Receive prompt
2. **HTTP Request** - Enhance prompt with SGR
3. **OpenAI** - Use enhanced prompt
4. **HTTP Request** - Post-analyze with SGR
5. **Respond to Webhook** - Return enriched result

## Setup in n8n

### 1. Using HTTP Request Node

1. Add **HTTP Request** node
2. Configure:
   - **Method**: POST
   - **URL**: `http://your-mcp-sgr:8080/v1/apply-sgr`
   - **Authentication**: None (or Generic Credential if using API key)
   - **Send Headers**: Add `Content-Type: application/json`
   - **Send Body**: JSON with your parameters
   - **Options**: 
     - Timeout: 30000 (30 seconds for complex analyses)
     - Continue On Fail: True (optional)

### 2. Using Credentials (Optional)

If your MCP-SGR server requires authentication:

1. Create **Generic Auth** credential
2. Set:
   - **Generic Auth Type**: Header Auth
   - **Header Name**: X-API-Key
   - **Header Value**: your-api-key

### 3. Error Handling

Add an **IF** node after HTTP Request to check:
```
{{ $json.error === undefined }}
```

## Import Ready Workflows

Copy and import these JSON workflows into n8n:

### Workflow 1: Task Analysis
```json
{
  "name": "MCP-SGR Task Analysis",
  "nodes": [
    {
      "parameters": {},
      "id": "manual-trigger",
      "name": "Manual Trigger",
      "type": "n8n-nodes-base.manualTrigger",
      "position": [250, 300]
    },
    {
      "parameters": {
        "method": "POST",
        "url": "http://localhost:8080/v1/apply-sgr",
        "sendHeaders": true,
        "headerParameters": {
          "parameters": [
            {
              "name": "Content-Type",
              "value": "application/json"
            }
          ]
        },
        "sendBody": true,
        "specifyBody": "json",
        "jsonBody": "{\n  \"task\": \"Design a caching strategy for our API\",\n  \"schema_type\": \"planning\",\n  \"budget\": \"full\"\n}"
      },
      "id": "http-sgr",
      "name": "SGR Analysis",
      "type": "n8n-nodes-base.httpRequest",
      "position": [450, 300]
    }
  ],
  "connections": {
    "Manual Trigger": {
      "main": [[{"node": "SGR Analysis", "type": "main", "index": 0}]]
    }
  }
}
```

### Workflow 2: Prompt Enhancement
```json
{
  "name": "MCP-SGR Prompt Enhancer",
  "nodes": [
    {
      "parameters": {
        "httpMethod": "POST",
        "path": "enhance-prompt"
      },
      "id": "webhook",
      "name": "Webhook",
      "type": "n8n-nodes-base.webhook",
      "position": [250, 300]
    },
    {
      "parameters": {
        "method": "POST",
        "url": "http://localhost:8080/v1/enhance-prompt",
        "sendBody": true,
        "specifyBody": "json",
        "jsonBody": "{\n  \"original_prompt\": \"{{ $json.prompt }}\",\n  \"target_model\": \"{{ $json.model || 'gpt-4' }}\"\n}"
      },
      "id": "enhance",
      "name": "Enhance Prompt",
      "type": "n8n-nodes-base.httpRequest",
      "position": [450, 300]
    },
    {
      "parameters": {
        "respondWith": "json",
        "responseBody": "={{ $json }}"
      },
      "id": "respond",
      "name": "Respond",
      "type": "n8n-nodes-base.respondToWebhook",
      "position": [650, 300]
    }
  ],
  "connections": {
    "Webhook": {
      "main": [[{"node": "Enhance Prompt", "type": "main", "index": 0}]]
    },
    "Enhance Prompt": {
      "main": [[{"node": "Respond", "type": "main", "index": 0}]]
    }
  }
}
```

## Tips and Best Practices

1. **Caching**: SGR caches results. Use consistent task descriptions for better cache hits.

2. **Budget Selection**:
   - `none`: Skip SGR analysis (passthrough)
   - `lite`: Quick analysis, good for production
   - `full`: Detailed analysis, good for debugging

3. **Schema Types**:
   - `auto`: Let SGR choose the best schema
   - `analysis`: For understanding problems
   - `planning`: For creating action plans
   - `decision`: For making choices
   - `code_generation`: For coding tasks
   - `summarization`: For condensing information

4. **Confidence Thresholds**:
   - `> 0.8`: High confidence, proceed automatically
   - `0.6 - 0.8`: Medium confidence, may need review
   - `< 0.6`: Low confidence, consider human review

5. **Performance**:
   - Use `budget: "lite"` for real-time workflows
   - Enable caching on MCP-SGR server
   - Set appropriate timeouts (30s for full analysis)

## Troubleshooting

### Connection Refused
- Check MCP-SGR server is running: `curl http://localhost:8080/health`
- Verify firewall rules
- Use correct hostname in Docker networks

### Timeout Errors
- Increase timeout in HTTP Request node
- Use `budget: "lite"` for faster responses
- Check server resources

### Low Confidence Results
- Try different schema_type
- Provide more context
- Use `budget: "full"` for complex tasks

## Support

- GitHub Issues: https://github.com/mcp-sgr/mcp-sgr/issues
- Documentation: https://docs.mcp-sgr.dev
- n8n Community: https://community.n8n.io