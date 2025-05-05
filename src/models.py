from pydantic import BaseModel, Field, create_model
from typing import Dict, Any, List, Optional

class ComplianceRequest(BaseModel):
    text: str
    
class ComplianceResponse(BaseModel):
    element: str
    parameters: Dict[str, Any]

class Query(BaseModel):
    text: str

class Relevance(BaseModel):
    relevance: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = Field(
        default=None, 
        description="Explanation for the relevance score"
    )

# base compliance class with common fields
class BaseRampCompliance(BaseModel):
    ramp_run: Optional[float] = 18000.0  
    ramp_landing_length: Optional[float] = 1500.0  
    ramp_width: Optional[float] = 1200.0 
    path_width: Optional[float] = 1200.0

# dynamic model that allows any field for ramp gradients
class DynamicRampCompliance(BaseRampCompliance):
    gradient_max_lengths: Dict[str, float] = {}

    class Config:
        extra = "ignore"  # This prevents extra fields from being included

    def dict(self, *args, **kwargs):
        result = {
            "ramp_run": self.ramp_run,
            "ramp_landing_length": self.ramp_landing_length,
            "ramp_width": self.ramp_width,
            "path_width": self.path_width,
            "gradient_max_lengths": self.gradient_max_lengths
        }
        return {k: v for k, v in result.items() if v is not None}

# legacy models kept for backward compatibility
class RampCompliance(BaseRampCompliance):
    ramp_length_1_12: Optional[float] = 6000.0 
    ramp_length_1_14: Optional[float] = 9000.0  
    ramp_length_1_15: Optional[float] = 11000.0
    ramp_length_1_20: Optional[float] = 15000.0  
    
class ResearchCompliance(BaseRampCompliance):
    ramp_length_1_12: Optional[float] = 6000.0 
    ramp_length_1_14: Optional[float] = 9000.0  
    ramp_length_1_15: Optional[float] = 11000.0
    ramp_length_1_22: Optional[float] = 15000.0  

class UdiRequest(BaseModel):
    text: str
    pdf_name: str  # name of specific UDI PDF to search (without extension)
    
class ClauseResponse(BaseModel):
    document: Optional[str] = None
    clause: Optional[str] = None
    section: Optional[str] = None
    full_citation: Optional[str] = None