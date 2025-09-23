# Database Operations Tool - Example Prompts

## Complete Prompts (Should Execute Successfully)

### 1. Create Savings Account
```
Create a new savings account called 'Emergency Fund' with a current balance of ₹50,000, target amount of ₹100,000, interest rate of 4.5%, and monthly contribution of ₹5,000
```

### 2. Create Asset
```
Add a new investment of ₹50,000 in mutual funds to my portfolio. The investment is called 'Nifty 50 Index Fund' with a current value of ₹50,000 and purchase date of today
```

### 3. Update Entity
```
Update my savings account with ID 'some-existing-id' to have a current balance of ₹75,000 and target amount of ₹150,000
```

### 4. Transfer Between Accounts
```
Transfer ₹25,000 from savings account 'source-account-id' to savings account 'destination-account-id'
```

## Incomplete Prompts (Should Ask for Clarification)

### 1. Missing Critical Parameters
```
Create a new savings account called 'Vacation Fund'
```
**Expected**: System should ask for current_balance, interest_rate, monthly_contribution

### 2. Missing Entity ID for Update
```
Update my emergency fund balance to ₹75,000
```
**Expected**: System should ask for entity_id or try to find the entity

### 3. Missing Entity ID for Delete
```
Delete my old savings account
```
**Expected**: System should ask for entity_id to identify which account

### 4. Missing Transfer Details
```
Transfer ₹10,000 from my emergency fund to my vacation fund
```
**Expected**: System should ask for specific entity IDs for the transfer

## How the System Handles Missing Parameters

### 1. Parameter Validation
- The system uses Pydantic models to validate parameters
- Critical parameters are defined in `config/intent_parameters.yaml`
- Missing critical parameters trigger clarification flow

### 2. Clarification Flow
- **Completeness Validator**: Checks if all critical parameters are present
- **Missing Parameter Analysis**: Uses LLM to identify what's missing
- **Clarification Question Generator**: Creates targeted questions
- **User Response Processor**: Parses user responses and updates parameters

### 3. Error Handling
- **Validation Errors**: Pydantic validation errors are caught and reported
- **Database Errors**: Foreign key constraints, schema mismatches are handled
- **Timeout Errors**: Long-running operations have timeout protection

### 4. Fallback Behavior
- If max clarification attempts are reached, the system may:
  - Use default values where possible
  - Return an error with specific missing parameters
  - Suggest alternative approaches

## Testing Commands

### Run Complete Test Suite
```bash
python test_langgraph_database_operations.py
```

### Test Individual Operations
```bash
# Start the API server first
python start_api.py

# Then test with curl or the test script
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a new savings account called Emergency Fund with current balance ₹50000",
    "userId": "cmfmk43mo0000tbz5dbkdpohl",
    "financialData": {}
  }'
```

## Expected System Behavior

### For Complete Prompts:
1. ✅ Intent classification succeeds
2. ✅ Parameter extraction finds all required fields
3. ✅ Database operation executes successfully
4. ✅ Returns success message with operation details

### For Incomplete Prompts:
1. ✅ Intent classification succeeds
2. ⚠️ Parameter extraction identifies missing fields
3. 🔄 System enters clarification flow
4. ❓ Asks targeted questions for missing parameters
5. 🔄 Waits for user response
6. ✅ Eventually executes operation or returns helpful error

### Error Scenarios:
1. **Invalid Entity ID**: Returns "Entity not found" error
2. **Foreign Key Violation**: Returns constraint violation error
3. **Schema Mismatch**: Returns column/field error
4. **Validation Error**: Returns parameter validation error
