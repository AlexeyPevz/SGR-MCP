"""RAG-specific reasoning schemas."""

from typing import List, Dict, Any, Optional
from .base import BaseSchema, SchemaField


class RAGAnalysisSchema(BaseSchema):
    """Schema for RAG pre-analysis phase."""
    
    def get_description(self) -> str:
        return "Structured analysis for RAG retrieval preparation"
    
    def get_fields(self) -> List[SchemaField]:
        return [
            # Query understanding
            SchemaField(
                name="query_analysis",
                type="object",
                required=True,
                description="Analysis of the user query"
            ),
            SchemaField(
                name="search_strategy",
                type="object", 
                required=True,
                description="Strategy for document retrieval"
            ),
            SchemaField(
                name="relevance_criteria",
                type="array",
                required=True,
                description="Criteria for assessing document relevance"
            ),
            SchemaField(
                name="expected_evidence",
                type="object",
                required=False,
                description="Expected types of evidence to find"
            )
        ]
    
    def get_examples(self) -> List[Dict[str, Any]]:
        return [{
            "query_analysis": {
                "intent": "Find information about RAG optimization techniques",
                "key_concepts": ["RAG", "optimization", "retrieval", "performance"],
                "ambiguities": ["Which aspect of optimization - speed, accuracy, or cost?"],
                "implicit_requirements": ["Recent techniques", "Practical implementation"]
            },
            "search_strategy": {
                "primary_queries": [
                    "RAG optimization techniques 2024",
                    "improving retrieval augmented generation performance"
                ],
                "expansion_terms": ["vector search", "reranking", "hybrid search"],
                "filters": {
                    "date_range": "last 2 years",
                    "source_type": ["research papers", "technical blogs"]
                }
            },
            "relevance_criteria": [
                "Contains specific optimization techniques",
                "Includes performance metrics or benchmarks",
                "Discusses practical implementation"
            ],
            "expected_evidence": {
                "techniques": ["query expansion", "reranking", "chunking strategies"],
                "metrics": ["latency", "recall", "precision"],
                "comparisons": "before/after optimization results"
            }
        }]
    
    def generate_prompt(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        prompt = f"""Analyze this query for RAG retrieval preparation:

Query: {task}

Provide a structured analysis following this schema:

1. Query Analysis:
   - What is the user's intent?
   - What are the key concepts?
   - What ambiguities need clarification?
   - What implicit requirements exist?

2. Search Strategy:
   - What primary search queries should be used?
   - What expansion terms could help?
   - What filters should be applied?

3. Relevance Criteria:
   - What makes a document relevant for this query?
   - What specific information should documents contain?

4. Expected Evidence:
   - What types of evidence would answer this query?
   - What metrics or data points are needed?
"""
        if context and context.get("domain"):
            prompt += f"\nDomain context: {context['domain']}"
        
        return prompt


class RAGValidationSchema(BaseSchema):
    """Schema for RAG post-retrieval validation."""
    
    def get_description(self) -> str:
        return "Validation of RAG retrieval results and coverage"
    
    def get_fields(self) -> List[SchemaField]:
        return [
            SchemaField(
                name="evidence_analysis",
                type="object",
                required=True,
                description="Analysis of retrieved evidence"
            ),
            SchemaField(
                name="claim_to_source_map",
                type="array",
                required=True,
                description="Mapping of claims to their sources"
            ),
            SchemaField(
                name="coverage_assessment",
                type="object",
                required=True,
                description="Assessment of query coverage"
            ),
            SchemaField(
                name="missing_evidence",
                type="array",
                required=False,
                description="Important information not found in sources"
            ),
            SchemaField(
                name="confidence_factors",
                type="object",
                required=True,
                description="Factors affecting answer confidence"
            )
        ]
    
    def get_examples(self) -> List[Dict[str, Any]]:
        return [{
            "evidence_analysis": {
                "total_sources": 5,
                "relevant_sources": 3,
                "source_quality": {
                    "high": 1,
                    "medium": 2,
                    "low": 0
                },
                "temporal_coverage": "2022-2024",
                "source_diversity": ["research papers", "technical blogs"]
            },
            "claim_to_source_map": [
                {
                    "claim": "Query expansion improves RAG recall by 15-20%",
                    "sources": ["doc_1", "doc_3"],
                    "confidence": "high",
                    "direct_quote": True
                },
                {
                    "claim": "Hybrid search outperforms pure vector search",
                    "sources": ["doc_2"],
                    "confidence": "medium",
                    "direct_quote": False
                }
            ],
            "coverage_assessment": {
                "query_aspects_covered": ["techniques", "performance metrics"],
                "query_aspects_missing": ["cost analysis", "implementation complexity"],
                "coverage_score": 0.75,
                "completeness": "partial"
            },
            "missing_evidence": [
                "Specific implementation examples",
                "Comparison with baseline methods",
                "Resource requirements"
            ],
            "confidence_factors": {
                "source_agreement": 0.8,
                "evidence_strength": "moderate",
                "gaps_impact": "minor",
                "overall_confidence": 0.72
            }
        }]
    
    def generate_prompt(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        prompt = f"""Validate the RAG retrieval results for this query:

Original Query: {task}

Retrieved Documents: {context.get('documents', 'Not provided')}

Generated Answer: {context.get('answer', 'Not provided')}

Provide a structured validation following this schema:

1. Evidence Analysis:
   - How many sources were retrieved and how many are relevant?
   - What is the quality distribution of sources?
   - What time period do the sources cover?

2. Claim-to-Source Mapping:
   - For each major claim in the answer, identify supporting sources
   - Rate confidence for each claim
   - Note if claims are direct quotes or inferences

3. Coverage Assessment:
   - Which aspects of the query are well-covered?
   - What is missing?
   - Overall coverage score (0-1)

4. Missing Evidence:
   - What important information was not found?
   - What follow-up searches might help?

5. Confidence Factors:
   - Do sources agree with each other?
   - How strong is the evidence?
   - What is the overall confidence level?
"""
        return prompt


# Register schemas
RAG_SCHEMAS = {
    "rag_analysis": RAGAnalysisSchema,
    "rag_validation": RAGValidationSchema
}