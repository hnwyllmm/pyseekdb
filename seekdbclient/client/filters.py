"""
Filter builder utilities for metadata and document filtering

Supports:
- Metadata filters: $eq, $lt, $gt, $lte, $gte, $ne, $in, $nin
- Logical operators: $or, $and, $not
- Document filters: $contains, $regex
"""
import re
from typing import Any, Dict, List, Optional, Tuple


class FilterBuilder:
    """Build SQL WHERE clauses from filter dictionaries"""
    
    # Comparison operators mapping
    COMPARISON_OPS = {
        "$eq": "=",
        "$lt": "<",
        "$gt": ">",
        "$lte": "<=",
        "$gte": ">=",
        "$ne": "!="
    }
    
    # Logical operators
    LOGICAL_OPS = ["$and", "$or", "$not"]
    
    # Document operators
    DOCUMENT_OPS = ["$contains", "$regex"]
    
    @staticmethod
    def build_metadata_filter(
        where: Dict[str, Any],
        metadata_column: str = "metadata"
    ) -> Tuple[str, List[Any]]:
        """
        Build WHERE clause for metadata filtering
        
        Args:
            where: Filter dictionary with operators like $eq, $lt, $gt, $lte, $gte, $ne, $in, $nin, $and, $or, $not
            metadata_column: Name of metadata column (default: "metadata")
            
        Returns:
            Tuple of (where_clause, params) for parameterized query
            
        Examples:
            where = {"age": {"$gte": 18}}
            -> ("JSON_EXTRACT(metadata, '$.age') >= %s", [18])
            
            where = {"$and": [{"age": {"$gte": 18}}, {"city": "Beijing"}]}
            -> ("(JSON_EXTRACT(metadata, '$.age') >= %s AND JSON_EXTRACT(metadata, '$.city') = %s)", [18, "Beijing"])
        """
        if not where:
            return "", []
        
        return FilterBuilder._build_condition(where, metadata_column)
    
    @staticmethod
    def build_document_filter(
        where_document: Dict[str, Any],
        document_column: str = "document"
    ) -> Tuple[str, List[Any]]:
        """
        Build WHERE clause for document filtering
        
        Args:
            where_document: Filter dictionary with $contains, $regex, $and, $or operators
            document_column: Name of document column (default: "document")
            
        Returns:
            Tuple of (where_clause, params) for parameterized query
            
        Examples:
            where_document = {"$contains": "python"}
            -> ("MATCH(document) AGAINST (%s IN NATURAL LANGUAGE MODE)", ["python"])
            
            where_document = {"$regex": "^hello.*world$"}
            -> ("document REGEXP %s", ["^hello.*world$"])
        """
        if not where_document:
            return "", []
        
        return FilterBuilder._build_document_condition(where_document, document_column)
    
    @staticmethod
    def _build_condition(
        condition: Dict[str, Any],
        metadata_column: str,
        params: Optional[List[Any]] = None
    ) -> Tuple[str, List[Any]]:
        """Recursively build condition from nested dictionary"""
        if params is None:
            params = []
        
        clauses = []
        
        for key, value in condition.items():
            if key in FilterBuilder.LOGICAL_OPS:
                # Handle logical operators
                if key == "$and":
                    sub_clauses = []
                    for sub_condition in value:
                        sub_clause, params = FilterBuilder._build_condition(sub_condition, metadata_column, params)
                        sub_clauses.append(sub_clause)
                    clauses.append(f"({' AND '.join(sub_clauses)})")
                
                elif key == "$or":
                    sub_clauses = []
                    for sub_condition in value:
                        sub_clause, params = FilterBuilder._build_condition(sub_condition, metadata_column, params)
                        sub_clauses.append(sub_clause)
                    clauses.append(f"({' OR '.join(sub_clauses)})")
                
                elif key == "$not":
                    sub_clause, params = FilterBuilder._build_condition(value, metadata_column, params)
                    clauses.append(f"NOT ({sub_clause})")
            
            elif isinstance(value, dict):
                # Handle comparison operators
                for op, op_value in value.items():
                    if op in FilterBuilder.COMPARISON_OPS:
                        sql_op = FilterBuilder.COMPARISON_OPS[op]
                        clauses.append(f"JSON_EXTRACT({metadata_column}, '$.{key}') {sql_op} %s")
                        params.append(op_value)
                    
                    elif op == "$in":
                        placeholders = ", ".join(["%s"] * len(op_value))
                        clauses.append(f"JSON_EXTRACT({metadata_column}, '$.{key}') IN ({placeholders})")
                        params.extend(op_value)
                    
                    elif op == "$nin":
                        placeholders = ", ".join(["%s"] * len(op_value))
                        clauses.append(f"JSON_EXTRACT({metadata_column}, '$.{key}') NOT IN ({placeholders})")
                        params.extend(op_value)
            
            else:
                # Direct equality comparison
                clauses.append(f"JSON_EXTRACT({metadata_column}, '$.{key}') = %s")
                params.append(value)
        
        where_clause = " AND ".join(clauses) if clauses else "1=1"
        return where_clause, params
    
    @staticmethod
    def _build_document_condition(
        condition: Dict[str, Any],
        document_column: str,
        params: Optional[List[Any]] = None
    ) -> Tuple[str, List[Any]]:
        """Build document filter condition"""
        if params is None:
            params = []
        
        clauses = []
        
        for key, value in condition.items():
            if key == "$contains":
                # Full-text search using MATCH AGAINST
                clauses.append(f"MATCH({document_column}) AGAINST (%s IN NATURAL LANGUAGE MODE)")
                params.append(value)
            
            elif key == "$regex":
                # Regular expression matching
                clauses.append(f"{document_column} REGEXP %s")
                params.append(value)
            
            elif key == "$and":
                sub_clauses = []
                for sub_condition in value:
                    sub_clause, params = FilterBuilder._build_document_condition(sub_condition, document_column, params)
                    sub_clauses.append(sub_clause)
                clauses.append(f"({' AND '.join(sub_clauses)})")
            
            elif key == "$or":
                sub_clauses = []
                for sub_condition in value:
                    sub_clause, params = FilterBuilder._build_document_condition(sub_condition, document_column, params)
                    sub_clauses.append(sub_clause)
                clauses.append(f"({' OR '.join(sub_clauses)})")
        
        where_clause = " AND ".join(clauses) if clauses else "1=1"
        return where_clause, params
    
    @staticmethod
    def combine_filters(
        metadata_filter: Tuple[str, List[Any]],
        document_filter: Tuple[str, List[Any]]
    ) -> Tuple[str, List[Any]]:
        """
        Combine metadata and document filters
        
        Args:
            metadata_filter: Tuple of (where_clause, params) for metadata
            document_filter: Tuple of (where_clause, params) for document
            
        Returns:
            Combined (where_clause, params)
        """
        meta_clause, meta_params = metadata_filter
        doc_clause, doc_params = document_filter
        
        clauses = []
        all_params = []
        
        if meta_clause:
            clauses.append(meta_clause)
            all_params.extend(meta_params)
        
        if doc_clause:
            clauses.append(doc_clause)
            all_params.extend(doc_params)
        
        if clauses:
            combined_clause = " AND ".join(clauses)
            return combined_clause, all_params
        else:
            return "", []

