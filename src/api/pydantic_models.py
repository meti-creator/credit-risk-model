from pydantic import BaseModel, Field

class CustomerPredictionInput(BaseModel):
    """
    Validates incoming request payload data structural types 
    before executing model array algebra mapping.
    """
    CustomerId: str = Field(..., description="Unique identification token for the customer profile")
    Amount: float = Field(..., description="The transaction volume amount value", gt=0)
    TransactionStartTime: str = Field(..., description="Timestamp string of the profile action (e.g., '2026-06-04 12:00:00')")
    
    # Configuration block to provide an interactive example in FastAPI's Swagger UI docs
    model_config = {
        "json_schema_extra": {
            "example": {
                "CustomerId": "Cust_Meti_99",
                "Amount": 2500.50,
                "TransactionStartTime": "2026-06-04 14:30:00"
            }
        }
    }

class PredictionResponse(BaseModel):
    """
    Enforces standardized JSON response output schema matrix fields.
    """
    customer_id: str
    risk_probability: float = Field(..., description="Calculated probability weight of high-risk operational classification")
    is_high_risk: int = Field(..., description="Binary target assignment prediction result (0 for Normal, 1 for High-Risk Proxy)")