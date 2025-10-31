# Sample Response for Advice Intent

This document shows sample responses that will be sent to the frontend after the graph executes for an **advice intent**.

## Example Request

```json
{
  "message": "I need financial advice on how to improve my savings rate",
  "financialData": {
    "user": {
      "id": "user_123",
      "name": "John Doe",
      "monthly_income": 50000,
      "monthly_expenses": 40000,
      "risk_profile": "moderate"
    },
    "dashboardMetrics": {
      "netWorth": 250000,
      "savingsRate": 0.2,
      "totalAssets": 300000,
      "totalLiabilities": 50000
    }
  },
  "userId": "user_123",
  "sessionId": "session_abc123"
}
```

---

## Response Format 1: Non-Streaming Endpoint (`POST /chat`)

This endpoint returns a complete response in a single JSON object after processing completes.

### Sample Response

```json
{
  "response": "HIGH RISK Strategy\n\nRecommendations:\n• Invest 60% of your disposable income (₹6,000/month) in equity mutual funds\n• Allocate ₹2,000/month to small-cap stocks for aggressive growth\n• Consider investing ₹1,000/month in cryptocurrency (Bitcoin/ETH) for diversification\n• Start a SIP in aggressive growth funds with ₹3,000/month\n• Explore alternative investments like peer-to-peer lending for 15-18% returns\n\nMathematical Calculations:\n- Current Monthly Disposable Income: ₹50,000 - ₹40,000 = ₹10,000\n- Investment Amount (60%): ₹10,000 × 0.60 = ₹6,000/month\n- Equity Investment: ₹6,000 × 0.50 = ₹3,000/month\n- Expected Annual Return: 18% (high-risk equity)\n- Year 1 Calculation: ₹36,000 × (1 + 0.18) = ₹36,000 × 1.18 = ₹42,480\n- Small-cap Stocks: ₹2,000 × 12 = ₹24,000/year\n- Expected Return (25%): ₹24,000 × 1.25 = ₹30,000\n- Crypto Investment: ₹1,000 × 12 = ₹12,000/year\n- Expected Return (30%, high volatility): ₹12,000 × 1.30 = ₹15,600\n- Total Expected Value After 1 Year: ₹42,480 + ₹30,000 + ₹15,600 = ₹88,080\n- Total Investment: ₹72,000\n- Net Gain: ₹88,080 - ₹72,000 = ₹16,080\n- ROI: (₹16,080 / ₹72,000) × 100 = 22.33%\n\nMEDIUM RISK Strategy\n\nRecommendations:\n• Allocate 40% of disposable income (₹4,000/month) to balanced mutual funds\n• Invest ₹2,000/month in large-cap equity funds for stability\n• Create an emergency fund with ₹2,000/month in liquid funds\n• Start a hybrid SIP with ₹2,000/month (60% equity, 40% debt)\n• Consider debt mutual funds for ₹1,000/month to balance risk\n\nMathematical Calculations:\n- Investment Amount (40%): ₹10,000 × 0.40 = ₹4,000/month\n- Balanced Funds (₹2,000/month): ₹24,000/year\n- Expected Return (12%): ₹24,000 × 1.12 = ₹26,880\n- Large-cap Equity (₹2,000/month): ₹24,000/year\n- Expected Return (14%): ₹24,000 × 1.14 = ₹27,360\n- Emergency Fund (Liquid): ₹24,000/year, return 6% = ₹25,440\n- Total Investment: ₹48,000\n- Total Expected Value: ₹26,880 + ₹27,360 + ₹25,440 = ₹79,680\n- Net Gain: ₹79,680 - ₹48,000 = ₹31,680\n- ROI: (₹31,680 / ₹48,000) × 100 = 66% (includes emergency fund which is more stable)\n- Annualized Return: 15.5%\n\nLOW RISK Strategy\n\nRecommendations:\n• Invest 30% of disposable income (₹3,000/month) in fixed deposits and bonds\n• Allocate ₹2,000/month to debt mutual funds for capital preservation\n• Create a systematic savings plan with ₹1,000/month in recurring deposits\n• Invest in government bonds and PPF with ₹2,000/month\n• Maintain high liquidity with ₹1,000/month in savings account\n\nMathematical Calculations:\n- Investment Amount (30%): ₹10,000 × 0.30 = ₹3,000/month\n- Fixed Deposits (₹2,000/month): ₹24,000/year at 7.5% interest\n- Calculation: ₹24,000 × (1 + 0.075) = ₹24,000 × 1.075 = ₹25,800\n- Debt Mutual Funds (₹1,000/month): ₹12,000/year at 8% return\n- Calculation: ₹12,000 × 1.08 = ₹12,960\n- Recurring Deposits (₹1,000/month): ₹12,000/year at 7% interest\n- Calculation: ₹12,000 × 1.07 = ₹12,840\n- Total Investment: ₹36,000\n- Total Expected Value: ₹25,800 + ₹12,960 + ₹12,840 = ₹51,600\n- Net Gain: ₹51,600 - ₹36,000 = ₹15,600\n- ROI: (₹15,600 / ₹36,000) × 100 = 43.33%\n- Annualized Return: 7.8%",
  "session_id": "session_abc123",
  "status": "success",
  "conversation_history": [
    {
      "role": "user",
      "content": "I need financial advice on how to improve my savings rate",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    {
      "role": "assistant",
      "content": "HIGH RISK Strategy\n\nRecommendations:\n• Invest 60% of your disposable income (₹6,000/month) in equity mutual funds\n• Allocate ₹2,000/month to small-cap stocks for aggressive growth\n• Consider investing ₹1,000/month in cryptocurrency (Bitcoin/ETH) for diversification\n• Start a SIP in aggressive growth funds with ₹3,000/month\n• Explore alternative investments like peer-to-peer lending for 15-18% returns\n\nMathematical Calculations:\n- Current Monthly Disposable Income: ₹50,000 - ₹40,000 = ₹10,000\n- Investment Amount (60%): ₹10,000 × 0.60 = ₹6,000/month\n- Equity Investment: ₹6,000 × 0.50 = ₹3,000/month\n- Expected Annual Return: 18% (high-risk equity)\n- Year 1 Calculation: ₹36,000 × (1 + 0.18) = ₹36,000 × 1.18 = ₹42,480\n- Small-cap Stocks: ₹2,000 × 12 = ₹24,000/year\n- Expected Return (25%): ₹24,000 × 1.25 = ₹30,000\n- Crypto Investment: ₹1,000 × 12 = ₹12,000/year\n- Expected Return (30%, high volatility): ₹12,000 × 1.30 = ₹15,600\n- Total Expected Value After 1 Year: ₹42,480 + ₹30,000 + ₹15,600 = ₹88,080\n- Total Investment: ₹72,000\n- Net Gain: ₹88,080 - ₹72,000 = ₹16,080\n- ROI: (₹16,080 / ₹72,000) × 100 = 22.33%\n\nMEDIUM RISK Strategy\n\nRecommendations:\n• Allocate 40% of disposable income (₹4,000/month) to balanced mutual funds\n• Invest ₹2,000/month in large-cap equity funds for stability\n• Create an emergency fund with ₹2,000/month in liquid funds\n• Start a hybrid SIP with ₹2,000/month (60% equity, 40% debt)\n• Consider debt mutual funds for ₹1,000/month to balance risk\n\nMathematical Calculations:\n- Investment Amount (40%): ₹10,000 × 0.40 = ₹4,000/month\n- Balanced Funds (₹2,000/month): ₹24,000/year\n- Expected Return (12%): ₹24,000 × 1.12 = ₹26,880\n- Large-cap Equity (₹2,000/month): ₹24,000/year\n- Expected Return (14%): ₹24,000 × 1.14 = ₹27,360\n- Emergency Fund (Liquid): ₹24,000/year, return 6% = ₹25,440\n- Total Investment: ₹48,000\n- Total Expected Value: ₹26,880 + ₹27,360 + ₹25,440 = ₹79,680\n- Net Gain: ₹79,680 - ₹48,000 = ₹31,680\n- ROI: (₹31,680 / ₹48,000) × 100 = 66% (includes emergency fund which is more stable)\n- Annualized Return: 15.5%\n\nLOW RISK Strategy\n\nRecommendations:\n• Invest 30% of disposable income (₹3,000/month) in fixed deposits and bonds\n• Allocate ₹2,000/month to debt mutual funds for capital preservation\n• Create a systematic savings plan with ₹1,000/month in recurring deposits\n• Invest in government bonds and PPF with ₹2,000/month\n• Maintain high liquidity with ₹1,000/month in savings account\n\nMathematical Calculations:\n- Investment Amount (30%): ₹10,000 × 0.30 = ₹3,000/month\n- Fixed Deposits (₹2,000/month): ₹24,000/year at 7.5% interest\n- Calculation: ₹24,000 × (1 + 0.075) = ₹24,000 × 1.075 = ₹25,800\n- Debt Mutual Funds (₹1,000/month): ₹12,000/year at 8% return\n- Calculation: ₹12,000 × 1.08 = ₹12,960\n- Recurring Deposits (₹1,000/month): ₹12,000/year at 7% interest\n- Calculation: ₹12,000 × 1.07 = ₹12,840\n- Total Investment: ₹36,000\n- Total Expected Value: ₹25,800 + ₹12,960 + ₹12,840 = ₹51,600\n- Net Gain: ₹51,600 - ₹36,000 = ₹15,600\n- ROI: (₹15,600 / ₹36,000) × 100 = 43.33%\n- Annualized Return: 7.8%",
      "timestamp": "2024-01-15T10:30:15Z"
    }
  ],
  "metadata": {
    "api_version": "0.1.0",
    "processing_time_ms": 3456.78,
    "intent": "advice",
    "intent_confidence": 0.92,
    "main_graph_status": "completed",
    "final_response_formatted": true,
    "conversation_turn_completed": true,
    "orchestration_completed": true,
    "final_response_generated": true,
    "tool_execution_results": {
      "advice": {
        "success": true,
        "data": {
          "advice": "HIGH RISK Strategy\n\nRecommendations:\n• Invest 60% of your disposable income (₹6,000/month) in equity mutual funds\n• Allocate ₹2,000/month to small-cap stocks for aggressive growth\n• Consider investing ₹1,000/month in cryptocurrency (Bitcoin/ETH) for diversification\n• Start a SIP in aggressive growth funds with ₹3,000/month\n• Explore alternative investments like peer-to-peer lending for 15-18% returns\n\nMEDIUM RISK Strategy\n\nRecommendations:\n• Allocate 40% of disposable income (₹4,000/month) to balanced mutual funds\n• Invest ₹2,000/month in large-cap equity funds for stability\n• Create an emergency fund with ₹2,000/month in liquid funds\n• Start a hybrid SIP with ₹2,000/month (60% equity, 40% debt)\n• Consider debt mutual funds for ₹1,000/month to balance risk\n\nLOW RISK Strategy\n\nRecommendations:\n• Invest 30% of disposable income (₹3,000/month) in fixed deposits and bonds\n• Allocate ₹2,000/month to debt mutual funds for capital preservation\n• Create a systematic savings plan with ₹1,000/month in recurring deposits\n• Invest in government bonds and PPF with ₹2,000/month\n• Maintain high liquidity with ₹1,000/month in savings account",
          "calculations": "HIGH RISK Calculations:\n- Current Monthly Disposable Income: ₹50,000 - ₹40,000 = ₹10,000\n- Investment Amount (60%): ₹10,000 × 0.60 = ₹6,000/month\n- Equity Investment: ₹6,000 × 0.50 = ₹3,000/month\n- Expected Annual Return: 18% (high-risk equity)\n- Year 1 Calculation: ₹36,000 × (1 + 0.18) = ₹36,000 × 1.18 = ₹42,480\n- Small-cap Stocks: ₹2,000 × 12 = ₹24,000/year\n- Expected Return (25%): ₹24,000 × 1.25 = ₹30,000\n- Crypto Investment: ₹1,000 × 12 = ₹12,000/year\n- Expected Return (30%, high volatility): ₹12,000 × 1.30 = ₹15,600\n- Total Expected Value After 1 Year: ₹42,480 + ₹30,000 + ₹15,600 = ₹88,080\n- Total Investment: ₹72,000\n- Net Gain: ₹88,080 - ₹72,000 = ₹16,080\n- ROI: (₹16,080 / ₹72,000) × 100 = 22.33%\n\nMEDIUM RISK Calculations:\n- Investment Amount (40%): ₹10,000 × 0.40 = ₹4,000/month\n- Balanced Funds (₹2,000/month): ₹24,000/year\n- Expected Return (12%): ₹24,000 × 1.12 = ₹26,880\n- Large-cap Equity (₹2,000/month): ₹24,000/year\n- Expected Return (14%): ₹24,000 × 1.14 = ₹27,360\n- Emergency Fund (Liquid): ₹24,000/year, return 6% = ₹25,440\n- Total Investment: ₹48,000\n- Total Expected Value: ₹26,880 + ₹27,360 + ₹25,440 = ₹79,680\n- Net Gain: ₹79,680 - ₹48,000 = ₹31,680\n- ROI: (₹31,680 / ₹48,000) × 100 = 66%\n- Annualized Return: 15.5%\n\nLOW RISK Calculations:\n- Investment Amount (30%): ₹10,000 × 0.30 = ₹3,000/month\n- Fixed Deposits (₹2,000/month): ₹24,000/year at 7.5% interest\n- Calculation: ₹24,000 × (1 + 0.075) = ₹24,000 × 1.075 = ₹25,800\n- Debt Mutual Funds (₹1,000/month): ₹12,000/year at 8% return\n- Calculation: ₹12,000 × 1.08 = ₹12,960\n- Recurring Deposits (₹1,000/month): ₹12,000/year at 7% interest\n- Calculation: ₹12,000 × 1.07 = ₹12,840\n- Total Investment: ₹36,000\n- Total Expected Value: ₹25,800 + ₹12,960 + ₹12,840 = ₹51,600\n- Net Gain: ₹51,600 - ₹36,000 = ₹15,600\n- ROI: (₹15,600 / ₹36,000) × 100 = 43.33%\n- Annualized Return: 7.8%",
          "full_response": "[Full LLM response with recommendations and calculations]",
          "user_query": "I need financial advice on how to improve my savings rate",
          "context_used": true,
          "user_id": "user_123",
          "risk_levels": ["high", "medium", "low"],
          "has_calculations": true
        },
        "tool_name": "advice",
        "execution_time_ms": 2345.67
      }
    },
    "execution_plan_used": {
      "steps": [
        {
          "step_id": "step_1",
          "tool_name": "advice",
          "parameters": {
            "user_query": "I need financial advice on how to improve my savings rate",
            "user_id": "user_123",
            "financial_data": {
              "user": {
                "id": "user_123",
                "name": "John Doe",
                "monthly_income": 50000,
                "monthly_expenses": 40000,
                "risk_profile": "moderate"
              },
              "dashboardMetrics": {
                "netWorth": 250000,
                "savingsRate": 0.2,
                "totalAssets": 300000,
                "totalLiabilities": 50000
              }
            }
          },
          "status": "completed",
          "result": "success"
        }
      ]
    }
  },
  "processing_time_ms": 3456.78
}
```

---

## Response Format 2: Streaming Endpoint (`POST /chat/stream`)

This endpoint streams Server-Sent Events (SSE) in real-time as the graph processes the request.

### Sample Stream Events

The response is sent as multiple SSE events, one per line with `data: ` prefix:

```
data: {"type": "connection", "session_id": "session_abc123", "status": "connected"}

data: {"type": "heartbeat", "message": "Initializing...", "timestamp": 1705315800.123}

data: {"type": "processing", "message": "Processing your request...", "stage": "user_input_processor"}

data: {"type": "processing", "message": "Understanding your intent...", "stage": "intent_classification_node"}

data: {"type": "processing", "message": "Routing to advice handler...", "stage": "decision_router"}

data: {"type": "processing", "message": "Planning execution...", "stage": "execution_planner"}

data: {"type": "processing", "message": "Generating personalized advice...", "stage": "tool_execution_node"}

data: {"type": "processing", "message": "Generating response...", "stage": "response_generation"}

data: {"type": "response_chunk", "content": "HIGH RISK Strategy", "is_final": false, "chunk_index": 0, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n\nRecommendations:", "is_final": false, "chunk_index": 1, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n• Invest 60% of your disposable income (₹6,000/month) in equity mutual funds", "is_final": false, "chunk_index": 2, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n• Allocate ₹2,000/month to small-cap stocks for aggressive growth", "is_final": false, "chunk_index": 3, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n• Consider investing ₹1,000/month in cryptocurrency (Bitcoin/ETH) for diversification", "is_final": false, "chunk_index": 4, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n• Start a SIP in aggressive growth funds with ₹3,000/month", "is_final": false, "chunk_index": 5, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n• Explore alternative investments like peer-to-peer lending for 15-18% returns", "is_final": false, "chunk_index": 6, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n\nMathematical Calculations:", "is_final": false, "chunk_index": 7, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n- Current Monthly Disposable Income: ₹50,000 - ₹40,000 = ₹10,000", "is_final": false, "chunk_index": 8, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n- Investment Amount (60%): ₹10,000 × 0.60 = ₹6,000/month", "is_final": false, "chunk_index": 9, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n- Equity Investment: ₹6,000 × 0.50 = ₹3,000/month", "is_final": false, "chunk_index": 10, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n- Expected Annual Return: 18% (high-risk equity)", "is_final": false, "chunk_index": 11, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n- Year 1 Calculation: ₹36,000 × (1 + 0.18) = ₹36,000 × 1.18 = ₹42,480", "is_final": false, "chunk_index": 12, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n- Small-cap Stocks: ₹2,000 × 12 = ₹24,000/year", "is_final": false, "chunk_index": 13, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n- Expected Return (25%): ₹24,000 × 1.25 = ₹30,000", "is_final": false, "chunk_index": 14, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n- Crypto Investment: ₹1,000 × 12 = ₹12,000/year", "is_final": false, "chunk_index": 15, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n- Expected Return (30%, high volatility): ₹12,000 × 1.30 = ₹15,600", "is_final": false, "chunk_index": 16, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n- Total Expected Value After 1 Year: ₹42,480 + ₹30,000 + ₹15,600 = ₹88,080", "is_final": false, "chunk_index": 17, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n- Total Investment: ₹72,000", "is_final": false, "chunk_index": 18, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n- Net Gain: ₹88,080 - ₹72,000 = ₹16,080", "is_final": false, "chunk_index": 19, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n- ROI: (₹16,080 / ₹72,000) × 100 = 22.33%", "is_final": false, "chunk_index": 20, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n\nMEDIUM RISK Strategy", "is_final": false, "chunk_index": 21, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n\nRecommendations:", "is_final": false, "chunk_index": 22, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n• Allocate 40% of disposable income (₹4,000/month) to balanced mutual funds", "is_final": false, "chunk_index": 23, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n• Invest ₹2,000/month in large-cap equity funds for stability", "is_final": false, "chunk_index": 24, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n• Create an emergency fund with ₹2,000/month in liquid funds", "is_final": false, "chunk_index": 25, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n• Start a hybrid SIP with ₹2,000/month (60% equity, 40% debt)", "is_final": false, "chunk_index": 26, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n• Consider debt mutual funds for ₹1,000/month to balance risk", "is_final": false, "chunk_index": 27, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n\nMathematical Calculations:", "is_final": false, "chunk_index": 28, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n- Investment Amount (40%): ₹10,000 × 0.40 = ₹4,000/month", "is_final": false, "chunk_index": 29, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n- Balanced Funds (₹2,000/month): ₹24,000/year", "is_final": false, "chunk_index": 30, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n- Expected Return (12%): ₹24,000 × 1.12 = ₹26,880", "is_final": false, "chunk_index": 31, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n- Large-cap Equity (₹2,000/month): ₹24,000/year", "is_final": false, "chunk_index": 32, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n- Expected Return (14%): ₹24,000 × 1.14 = ₹27,360", "is_final": false, "chunk_index": 33, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n- Emergency Fund (Liquid): ₹24,000/year, return 6% = ₹25,440", "is_final": false, "chunk_index": 34, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n- Total Investment: ₹48,000", "is_final": false, "chunk_index": 35, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n- Total Expected Value: ₹26,880 + ₹27,360 + ₹25,440 = ₹79,680", "is_final": false, "chunk_index": 36, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n- Net Gain: ₹79,680 - ₹48,000 = ₹31,680", "is_final": false, "chunk_index": 37, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n- ROI: (₹31,680 / ₹48,000) × 100 = 66%", "is_final": false, "chunk_index": 38, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n- Annualized Return: 15.5%", "is_final": false, "chunk_index": 39, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n\nLOW RISK Strategy", "is_final": false, "chunk_index": 40, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n\nRecommendations:", "is_final": false, "chunk_index": 41, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n• Invest 30% of disposable income (₹3,000/month) in fixed deposits and bonds", "is_final": false, "chunk_index": 42, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n• Allocate ₹2,000/month to debt mutual funds for capital preservation", "is_final": false, "chunk_index": 43, "total_chunks": 45}

data: {"type": "response_chunk", "content": "\n• Create a systematic savings plan with ₹1,000/month in recurring deposits\n• Invest in government bonds and PPF with ₹2,000/month\n• Maintain high liquidity with ₹1,000/month in savings account\n\nMathematical Calculations:\n- Investment Amount (30%): ₹10,000 × 0.30 = ₹3,000/month\n- Fixed Deposits (₹2,000/month): ₹24,000/year at 7.5% interest\n- Calculation: ₹24,000 × (1 + 0.075) = ₹24,000 × 1.075 = ₹25,800\n- Debt Mutual Funds (₹1,000/month): ₹12,000/year at 8% return\n- Calculation: ₹12,000 × 1.08 = ₹12,960\n- Recurring Deposits (₹1,000/month): ₹12,000/year at 7% interest\n- Calculation: ₹12,000 × 1.07 = ₹12,840\n- Total Investment: ₹36,000\n- Total Expected Value: ₹25,800 + ₹12,960 + ₹12,840 = ₹51,600\n- Net Gain: ₹51,600 - ₹36,000 = ₹15,600\n- ROI: (₹15,600 / ₹36,000) × 100 = 43.33%\n- Annualized Return: 7.8%", "is_final": true, "chunk_index": 44, "total_chunks": 45}

data: {"type": "response_complete", "session_id": "session_abc123", "metadata": {"api_version": "0.1.0", "processing_time_seconds": 3.456, "total_chunks": 45, "session_saved": true, "intent": "advice", "intent_confidence": 0.92, "main_graph_status": "completed", "final_response_formatted": true, "conversation_turn_completed": true, "orchestration_completed": true, "final_response_generated": true}}

data: {"type": "stream_complete", "session_id": "session_abc123", "total_time": 3.456}
```

---

## Event Types in Streaming Response

### 1. Connection Event
```json
{
  "type": "connection",
  "session_id": "session_abc123",
  "status": "connected"
}
```

### 2. Heartbeat Events
```json
{
  "type": "heartbeat",
  "message": "Still processing...",
  "stage": "graph_execution",
  "timestamp": 1705315800.123,
  "elapsed_time": 2.5
}
```

### 3. Processing Status Events
```json
{
  "type": "processing",
  "message": "Processing your intent...",
  "stage": "intent_classification_node"
}
```

### 4. Response Chunks
```json
{
  "type": "response_chunk",
  "content": "HIGH RISK Strategy",
  "is_final": false,
  "chunk_index": 0,
  "total_chunks": 45
}
```

### 5. Response Complete Event
```json
{
  "type": "response_complete",
  "session_id": "session_abc123",
  "metadata": {
    "processing_time_seconds": 3.456,
    "total_chunks": 45,
    "session_saved": true,
    "intent": "advice",
    "intent_confidence": 0.92,
    "main_graph_status": "completed",
    "final_response_formatted": true,
    "conversation_turn_completed": true,
    "orchestration_completed": true,
    "final_response_generated": true
  }
}
```

### 6. Stream Complete Event
```json
{
  "type": "stream_complete",
  "session_id": "session_abc123",
  "total_time": 3.456
}
```

### 7. Error Events
```json
{
  "type": "error",
  "message": "Sorry, I encountered an error: [error details]",
  "error_code": "STREAMING_ERROR",
  "details": {
    "error": "[error message]",
    "processing_time_seconds": 1.234
  }
}
```

---

## Key Features of Advice Response

1. **Three-Level Risk Strategy**: The advice includes recommendations for HIGH, MEDIUM, and LOW risk levels
2. **Detailed Calculations**: Each strategy includes step-by-step mathematical calculations showing:
   - How amounts are calculated
   - Expected returns
   - ROI percentages
   - Total investment vs expected value

3. **Personalized Recommendations**: Based on:
   - User's monthly income and expenses
   - Current financial profile
   - Risk tolerance (if provided)

4. **Structured Format**: The response is organized with:
   - Clear section headers for each risk level
   - Bullet-point recommendations
   - Detailed calculation breakdowns

---

## Frontend Integration Notes

### For Non-Streaming Endpoint:
- Wait for the complete HTTP response
- Extract `response` field for display
- Use `metadata` for UI enhancements (progress indicators, confidence scores)
- Update conversation history with `conversation_history` array

### For Streaming Endpoint:
- Use EventSource API or fetch with streaming parser
- Accumulate `response_chunk` events to build the full response
- Show progress using `processing` events
- Update UI in real-time as chunks arrive
- Handle `response_complete` to finalize the response
- Display metadata from `response_complete` event

---

## Error Handling

If an error occurs during advice generation, the response will include:

```json
{
  "response": "I'm sorry, I encountered an error while generating your financial advice.",
  "status": "error",
  "metadata": {
    "error": "Error message",
    "error_code": "ADVICE_GENERATION_ERROR",
    "processing_time_ms": 1234.56
  }
}
```

Or in streaming mode:

```json
{
  "type": "error",
  "message": "Sorry, I encountered an error while generating your financial advice.",
  "error_code": "ADVICE_GENERATION_ERROR",
  "details": {
    "error": "Error details here",
    "processing_time_seconds": 1.234
  }
}
```

