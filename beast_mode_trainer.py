#!/usr/bin/env python3
"""
BEAST MODE Vanna.AI Training System
Generates HUNDREDS of high-quality training examples for maximum accuracy
"""

import os
import json
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import sqlalchemy
from sqlalchemy import inspect, text
import pandas as pd

# Import Vanna modules
from vanna.openai.openai_chat import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore

class BeastModeVannaTrainer(ChromaDB_VectorStore, OpenAI_Chat):
    """
    BEAST MODE Vanna implementation - generates 500+ training examples
    """
    
    def __init__(self, config=None):
        # Ensure we have a path for persistent storage
        if config and 'path' not in config:
            config['path'] = str(Path.cwd() / "vanna_storage" / "default_model")
        elif not config:
            config = {'path': str(Path.cwd() / "vanna_storage" / "default_model")}
            
        # Create the directory if it doesn't exist
        Path(config['path']).mkdir(parents=True, exist_ok=True)
        
        # Initialize parent classes with config
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)
        
        # Store config for later use
        self.config = config
        self.model_path = Path(config['path'])
        
        print("🔥 BEAST MODE Vanna trainer initialized - preparing for MASSIVE training!")

    def is_trained(self) -> bool:
        """Check if the model has been trained by looking for ChromaDB files"""
        chroma_db_file = self.model_path / "chroma.sqlite3"
        return chroma_db_file.exists()

    def beast_mode_training(self, engine, db_type: str, batch_size: int = 50) -> Dict[str, Any]:
        """
        BEAST MODE training - generates 500+ examples
        """
        print("🔥 STARTING BEAST MODE TRAINING 🔥")
        print("🎯 Target: Generate 500+ high-quality training examples")
        
        # Extract comprehensive database information
        inspector = inspect(engine)
        tables_info = self._extract_comprehensive_schema(engine, db_type, inspector)
        
        # Generate MASSIVE amounts of training data
        all_examples = []
        
        # 1. Enhanced DDL with detailed comments
        print("📋 Generating enhanced DDL statements...")
        ddl_examples = self._generate_enhanced_ddl(tables_info, db_type)
        all_examples.extend(ddl_examples)
        print(f"   Generated {len(ddl_examples)} DDL examples")
        
        # 2. Basic query patterns (simple SELECTs)
        print("🔍 Generating basic query patterns...")
        basic_queries = self._generate_basic_queries(tables_info, db_type)
        all_examples.extend(basic_queries)
        print(f"   Generated {len(basic_queries)} basic query examples")
        
        # 3. Advanced JOIN patterns
        print("🔗 Generating JOIN patterns...")
        join_queries = self._generate_join_patterns(tables_info, db_type, inspector)
        all_examples.extend(join_queries)
        print(f"   Generated {len(join_queries)} JOIN examples")
        
        # 4. Aggregation and GROUP BY patterns
        print("📊 Generating aggregation patterns...")
        agg_queries = self._generate_aggregation_patterns(tables_info, db_type)
        all_examples.extend(agg_queries)
        print(f"   Generated {len(agg_queries)} aggregation examples")
        
        # 5. Complex WHERE clause patterns
        print("🎯 Generating complex WHERE patterns...")
        where_queries = self._generate_where_patterns(tables_info, db_type)
        all_examples.extend(where_queries)
        print(f"   Generated {len(where_queries)} WHERE clause examples")
        
        # 6. Date and time patterns
        print("📅 Generating date/time patterns...")
        date_queries = self._generate_date_patterns(tables_info, db_type)
        all_examples.extend(date_queries)
        print(f"   Generated {len(date_queries)} date/time examples")
        
        # 7. Subquery patterns
        print("🔄 Generating subquery patterns...")
        subquery_patterns = self._generate_subquery_patterns(tables_info, db_type)
        all_examples.extend(subquery_patterns)
        print(f"   Generated {len(subquery_patterns)} subquery examples")
        
        # 8. Business logic patterns
        print("💼 Generating business logic patterns...")
        business_patterns = self._generate_business_patterns(tables_info, db_type)
        all_examples.extend(business_patterns)
        print(f"   Generated {len(business_patterns)} business logic examples")
        
        # 9. Window function patterns (if supported)
        print("🪟 Generating window function patterns...")
        window_patterns = self._generate_window_patterns(tables_info, db_type)
        all_examples.extend(window_patterns)
        print(f"   Generated {len(window_patterns)} window function examples")
        
        # 10. Real data samples for context
        print("🎲 Sampling real data for context...")
        data_samples = self._sample_real_data(engine, tables_info, db_type)
        all_examples.extend(data_samples)
        print(f"   Generated {len(data_samples)} real data examples")
        
        print(f"🔥 BEAST MODE: Generated {len(all_examples)} training examples!")
        
        # Train in batches
        stats = self._train_in_batches(all_examples, batch_size)
        
        # Save comprehensive metadata
        metadata = {
            "training_mode": "BEAST_MODE",
            "total_examples": len(all_examples),
            "db_type": db_type,
            "tables_count": len(tables_info),
            "training_stats": stats,
            "trained_at": str(pd.Timestamp.now())
        }
        self.save_training_metadata(metadata)
        
        print("🎉 BEAST MODE TRAINING COMPLETE!")
        print(f"📊 Total examples: {len(all_examples)}")
        print(f"🎯 Your model is now a TRUE BEAST!")
        
        return stats

    def _extract_comprehensive_schema(self, engine, db_type: str, inspector) -> Dict[str, Any]:
        """Extract comprehensive schema information"""
        tables_info = {}
        
        if db_type == 'postgresql':
            schemas = [s for s in inspector.get_schema_names() if s not in ('pg_catalog', 'information_schema')]
            schema = schemas[0] if schemas else 'public'
        else:
            schema = None
        
        tables = inspector.get_table_names(schema=schema)
        
        for table in tables:
            columns = inspector.get_columns(table, schema=schema)
            
            # Categorize columns by type
            numeric_cols = []
            text_cols = []
            date_cols = []
            boolean_cols = []
            
            for col in columns:
                col_type = str(col['type']).lower()
                if any(t in col_type for t in ['int', 'float', 'decimal', 'numeric']):
                    numeric_cols.append(col['name'])
                elif any(t in col_type for t in ['varchar', 'text', 'char']):
                    text_cols.append(col['name'])
                elif any(t in col_type for t in ['date', 'time', 'timestamp']):
                    date_cols.append(col['name'])
                elif 'bool' in col_type:
                    boolean_cols.append(col['name'])
            
            # Get relationships
            try:
                foreign_keys = inspector.get_foreign_keys(table, schema=schema)
                primary_keys = inspector.get_pk_constraint(table, schema=schema)
            except:
                foreign_keys = []
                primary_keys = {'constrained_columns': []}
            
            tables_info[table] = {
                'schema': schema,
                'columns': columns,
                'numeric_cols': numeric_cols,
                'text_cols': text_cols,
                'date_cols': date_cols,
                'boolean_cols': boolean_cols,
                'foreign_keys': foreign_keys,
                'primary_keys': primary_keys.get('constrained_columns', []),
                'full_name': f"{schema}.{table}" if schema else table
            }
        
        return tables_info

    def _generate_enhanced_ddl(self, tables_info: Dict, db_type: str) -> List[Dict]:
        """Generate enhanced DDL with detailed comments"""
        examples = []
        
        for table, info in tables_info.items():
            # Basic DDL
            columns_ddl = []
            comments = []
            
            for col in info['columns']:
                col_type = str(col['type'])
                nullable = "" if col.get('nullable', True) else " NOT NULL"
                column_def = f"{col['name']} {col_type}{nullable}"
                columns_ddl.append(column_def)
                
                # Add column comments
                col_name = col['name']
                if col_name in info['numeric_cols']:
                    comments.append(f"-- {col_name}: Numeric field for calculations and aggregations")
                elif col_name in info['text_cols']:
                    comments.append(f"-- {col_name}: Text field for search and filtering")
                elif col_name in info['date_cols']:
                    comments.append(f"-- {col_name}: Date/time field for temporal analysis")
            
            ddl = f"CREATE TABLE {table} (\n  " + ",\n  ".join(columns_ddl) + "\n);"
            
            examples.append({
                'type': 'ddl',
                'content': ddl + "\n" + "\n".join(comments)
            })
            
            # Add table documentation
            examples.append({
                'type': 'documentation',
                'content': f"Table {table} contains {len(info['columns'])} columns including {len(info['numeric_cols'])} numeric fields, {len(info['text_cols'])} text fields, and {len(info['date_cols'])} date fields."
            })
        
        return examples

    def _generate_basic_queries(self, tables_info: Dict, db_type: str) -> List[Dict]:
        """Generate basic query patterns"""
        examples = []
        limit_clause = "TOP 10" if db_type == 'mssql' else "LIMIT 10"
        
        for table, info in tables_info.items():
            table_name = info['full_name']
            
            # Basic SELECT patterns
            queries = [
                f"SELECT * FROM {table_name} {limit_clause};",
                f"SELECT COUNT(*) FROM {table_name};",
                f"SELECT COUNT(*) as total_records FROM {table_name};",
            ]
            
            # Column-specific queries
            if info['columns']:
                # First few columns
                cols = [c['name'] for c in info['columns'][:3]]
                if cols:
                    queries.append(f"SELECT {', '.join(cols)} FROM {table_name} {limit_clause};")
                
                # All column names
                all_cols = [c['name'] for c in info['columns']]
                if len(all_cols) > 3:
                    queries.append(f"SELECT {', '.join(all_cols[:5])} FROM {table_name} {limit_clause};")
                
                # Specific column types
                if info['numeric_cols']:
                    for col in info['numeric_cols'][:2]:  # First 2 numeric columns
                        queries.extend([
                            f"SELECT {col}, COUNT(*) FROM {table_name} GROUP BY {col} ORDER BY COUNT(*) DESC {limit_clause};",
                            f"SELECT AVG({col}) as avg_{col} FROM {table_name};",
                            f"SELECT MIN({col}) as min_{col}, MAX({col}) as max_{col} FROM {table_name};",
                            f"SELECT SUM({col}) as total_{col} FROM {table_name};"
                        ])
                
                if info['text_cols']:
                    for col in info['text_cols'][:2]:  # First 2 text columns
                        queries.extend([
                            f"SELECT DISTINCT {col} FROM {table_name} {limit_clause};",
                            f"SELECT {col}, COUNT(*) FROM {table_name} GROUP BY {col} ORDER BY COUNT(*) DESC {limit_clause};",
                            f"SELECT {col} FROM {table_name} WHERE {col} IS NOT NULL {limit_clause};"
                        ])
                
                if info['date_cols']:
                    for col in info['date_cols']:
                        queries.extend([
                            f"SELECT {col} FROM {table_name} ORDER BY {col} DESC {limit_clause};",
                            f"SELECT DATE({col}) as date, COUNT(*) FROM {table_name} GROUP BY DATE({col}) ORDER BY date DESC {limit_clause};",
                            f"SELECT {col} FROM {table_name} WHERE {col} IS NOT NULL ORDER BY {col} {limit_clause};"
                        ])
            
            for query in queries:
                examples.append({
                    'type': 'sql',
                    'content': query
                })
        
        return examples

    def _generate_join_patterns(self, tables_info: Dict, db_type: str, inspector) -> List[Dict]:
        """Generate comprehensive JOIN patterns"""
        examples = []
        limit_clause = "TOP 20" if db_type == 'mssql' else "LIMIT 20"
        
        # Find all possible relationships
        for table, info in tables_info.items():
            for fk in info['foreign_keys']:
                if 'constrained_columns' in fk and 'referred_table' in fk:
                    ref_table = fk['referred_table']
                    if ref_table in tables_info:
                        local_col = fk['constrained_columns'][0]
                        ref_col = fk['referred_columns'][0]
                        
                        table_name = info['full_name']
                        ref_table_name = tables_info[ref_table]['full_name']
                        
                        # Various JOIN patterns
                        join_queries = [
                            f"SELECT t1.*, t2.* FROM {table_name} t1 JOIN {ref_table_name} t2 ON t1.{local_col} = t2.{ref_col} {limit_clause};",
                            f"SELECT t1.*, t2.* FROM {table_name} t1 LEFT JOIN {ref_table_name} t2 ON t1.{local_col} = t2.{ref_col} {limit_clause};",
                            f"SELECT t1.*, t2.* FROM {table_name} t1 RIGHT JOIN {ref_table_name} t2 ON t1.{local_col} = t2.{ref_col} {limit_clause};",
                            f"SELECT COUNT(*) as total FROM {table_name} t1 JOIN {ref_table_name} t2 ON t1.{local_col} = t2.{ref_col};",
                        ]
                        
                        # Add specific column joins
                        if tables_info[ref_table]['text_cols']:
                            ref_text_col = tables_info[ref_table]['text_cols'][0]
                            join_queries.extend([
                                f"SELECT t1.{local_col}, t2.{ref_text_col}, COUNT(*) FROM {table_name} t1 JOIN {ref_table_name} t2 ON t1.{local_col} = t2.{ref_col} GROUP BY t1.{local_col}, t2.{ref_text_col} {limit_clause};",
                                f"SELECT t2.{ref_text_col}, COUNT(t1.{local_col}) as count FROM {table_name} t1 RIGHT JOIN {ref_table_name} t2 ON t1.{local_col} = t2.{ref_col} GROUP BY t2.{ref_text_col} ORDER BY count DESC {limit_clause};"
                            ])
                        
                        if tables_info[ref_table]['numeric_cols']:
                            ref_num_col = tables_info[ref_table]['numeric_cols'][0]
                            join_queries.extend([
                                f"SELECT t1.{local_col}, AVG(t2.{ref_num_col}) as avg_value FROM {table_name} t1 JOIN {ref_table_name} t2 ON t1.{local_col} = t2.{ref_col} GROUP BY t1.{local_col} {limit_clause};",
                                f"SELECT t1.*, t2.{ref_num_col} FROM {table_name} t1 JOIN {ref_table_name} t2 ON t1.{local_col} = t2.{ref_col} WHERE t2.{ref_num_col} > 0 {limit_clause};"
                            ])
                        
                        for query in join_queries:
                            examples.append({
                                'type': 'sql',
                                'content': query
                            })
        
        return examples

    def _generate_aggregation_patterns(self, tables_info: Dict, db_type: str) -> List[Dict]:
        """Generate comprehensive aggregation patterns"""
        examples = []
        limit_clause = "TOP 15" if db_type == 'mssql' else "LIMIT 15"
        
        for table, info in tables_info.items():
            table_name = info['full_name']
            
            if info['numeric_cols']:
                for numeric_col in info['numeric_cols']:
                    # Single column aggregations
                    agg_queries = [
                        f"SELECT COUNT({numeric_col}) as count, AVG({numeric_col}) as average, SUM({numeric_col}) as total FROM {table_name};",
                        f"SELECT MIN({numeric_col}) as minimum, MAX({numeric_col}) as maximum FROM {table_name};",
                    ]
                    
                    # Group by text columns
                    if info['text_cols']:
                        for group_col in info['text_cols']:
                            agg_queries.extend([
                                f"SELECT {group_col}, SUM({numeric_col}) as total FROM {table_name} GROUP BY {group_col} ORDER BY total DESC {limit_clause};",
                                f"SELECT {group_col}, AVG({numeric_col}) as average FROM {table_name} GROUP BY {group_col} ORDER BY average DESC {limit_clause};",
                                f"SELECT {group_col}, COUNT(*) as count, MIN({numeric_col}) as min_val, MAX({numeric_col}) as max_val FROM {table_name} GROUP BY {group_col} ORDER BY count DESC {limit_clause};",
                                f"SELECT {group_col}, SUM({numeric_col}) as total, AVG({numeric_col}) as avg FROM {table_name} GROUP BY {group_col} HAVING COUNT(*) > 1 ORDER BY total DESC {limit_clause};",
                                f"SELECT {group_col}, COUNT(DISTINCT {numeric_col}) as unique_values FROM {table_name} GROUP BY {group_col} {limit_clause};"
                            ])
                    
                    # Group by date columns
                    if info['date_cols']:
                        for date_col in info['date_cols']:
                            agg_queries.extend([
                                f"SELECT DATE({date_col}) as date, SUM({numeric_col}) as daily_total FROM {table_name} GROUP BY DATE({date_col}) ORDER BY date DESC {limit_clause};",
                                f"SELECT EXTRACT(MONTH FROM {date_col}) as month, AVG({numeric_col}) as monthly_avg FROM {table_name} GROUP BY EXTRACT(MONTH FROM {date_col}) ORDER BY month {limit_clause};"
                            ])
                    
                    for query in agg_queries:
                        examples.append({
                            'type': 'sql',
                            'content': query
                        })
        
        return examples

    def _generate_where_patterns(self, tables_info: Dict, db_type: str) -> List[Dict]:
        """Generate complex WHERE clause patterns"""
        examples = []
        limit_clause = "TOP 10" if db_type == 'mssql' else "LIMIT 10"
        
        for table, info in tables_info.items():
            table_name = info['full_name']
            
            where_queries = []
            
            # Numeric WHERE clauses
            if info['numeric_cols']:
                for col in info['numeric_cols']:
                    where_queries.extend([
                        f"SELECT * FROM {table_name} WHERE {col} > 0 {limit_clause};",
                        f"SELECT * FROM {table_name} WHERE {col} < 0 {limit_clause};",
                        f"SELECT * FROM {table_name} WHERE {col} BETWEEN 1 AND 100 {limit_clause};",
                        f"SELECT * FROM {table_name} WHERE {col} IN (1, 2, 3, 4, 5) {limit_clause};",
                        f"SELECT * FROM {table_name} WHERE {col} IS NOT NULL ORDER BY {col} DESC {limit_clause};",
                        f"SELECT * FROM {table_name} WHERE {col} >= 10 AND {col} <= 100 {limit_clause};",
                        f"SELECT * FROM {table_name} WHERE {col} NOT IN (0, -1) {limit_clause};"
                    ])
            
            # Text WHERE clauses  
            if info['text_cols']:
                for col in info['text_cols']:
                    where_queries.extend([
                        f"SELECT * FROM {table_name} WHERE {col} LIKE '%test%' {limit_clause};",
                        f"SELECT * FROM {table_name} WHERE {col} LIKE 'A%' {limit_clause};",
                        f"SELECT * FROM {table_name} WHERE {col} IS NOT NULL AND {col} != '' {limit_clause};",
                        f"SELECT * FROM {table_name} WHERE LENGTH({col}) > 5 {limit_clause};",
                        f"SELECT * FROM {table_name} WHERE {col} ILIKE '%example%' {limit_clause};",
                        f"SELECT * FROM {table_name} WHERE {col} NOT LIKE '%test%' {limit_clause};"
                    ])
            
            # Date WHERE clauses
            if info['date_cols']:
                for col in info['date_cols']:
                    where_queries.extend([
                        f"SELECT * FROM {table_name} WHERE {col} >= CURRENT_DATE - INTERVAL '30 days' {limit_clause};",
                        f"SELECT * FROM {table_name} WHERE {col} >= CURRENT_DATE - INTERVAL '7 days' {limit_clause};",
                        f"SELECT * FROM {table_name} WHERE EXTRACT(YEAR FROM {col}) = 2024 {limit_clause};",
                        f"SELECT * FROM {table_name} WHERE EXTRACT(MONTH FROM {col}) = 12 {limit_clause};",
                        f"SELECT * FROM {table_name} WHERE {col} IS NOT NULL ORDER BY {col} DESC {limit_clause};",
                        f"SELECT * FROM {table_name} WHERE {col} BETWEEN '2024-01-01' AND '2024-12-31' {limit_clause};"
                    ])
            
            # Combined WHERE clauses
            if len(info['columns']) >= 2:
                col1 = info['columns'][0]['name']
                col2 = info['columns'][1]['name']
                where_queries.extend([
                    f"SELECT * FROM {table_name} WHERE {col1} IS NOT NULL AND {col2} IS NOT NULL {limit_clause};",
                    f"SELECT * FROM {table_name} WHERE {col1} IS NOT NULL OR {col2} IS NOT NULL {limit_clause};"
                ])
                
                # Numeric combinations
                if col1 in info['numeric_cols'] and col2 in info['numeric_cols']:
                    where_queries.extend([
                        f"SELECT * FROM {table_name} WHERE {col1} > {col2} {limit_clause};",
                        f"SELECT * FROM {table_name} WHERE {col1} + {col2} > 0 {limit_clause};"
                    ])
            
            for query in where_queries:
                examples.append({
                    'type': 'sql',
                    'content': query
                })
        
        return examples

    def _generate_date_patterns(self, tables_info: Dict, db_type: str) -> List[Dict]:
        """Generate date/time specific patterns"""
        examples = []
        limit_clause = "TOP 10" if db_type == 'mssql' else "LIMIT 10"
        
        for table, info in tables_info.items():
            if not info['date_cols']:
                continue
                
            table_name = info['full_name']
            
            for date_col in info['date_cols']:
                date_queries = [
                    f"SELECT DATE_TRUNC('month', {date_col}) as month, COUNT(*) FROM {table_name} GROUP BY month ORDER BY month DESC {limit_clause};",
                    f"SELECT DATE_TRUNC('week', {date_col}) as week, COUNT(*) FROM {table_name} GROUP BY week ORDER BY week DESC {limit_clause};",
                    f"SELECT EXTRACT(DOW FROM {date_col}) as day_of_week, COUNT(*) FROM {table_name} GROUP BY day_of_week ORDER BY day_of_week;",
                    f"SELECT EXTRACT(HOUR FROM {date_col}) as hour, COUNT(*) FROM {table_name} GROUP BY hour ORDER BY hour;",
                    f"SELECT * FROM {table_name} WHERE {date_col} >= CURRENT_DATE - INTERVAL '7 days' ORDER BY {date_col} DESC {limit_clause};",
                    f"SELECT * FROM {table_name} WHERE {date_col} >= CURRENT_DATE - INTERVAL '1 month' ORDER BY {date_col} DESC {limit_clause};",
                    f"SELECT DATE({date_col}) as date, COUNT(*) as daily_count FROM {table_name} GROUP BY DATE({date_col}) ORDER BY date DESC {limit_clause};",
                    f"SELECT EXTRACT(YEAR FROM {date_col}) as year, COUNT(*) FROM {table_name} GROUP BY year ORDER BY year DESC;",
                    f"SELECT * FROM {table_name} WHERE DATE({date_col}) = CURRENT_DATE {limit_clause};",
                    f"SELECT * FROM {table_name} WHERE {date_col} BETWEEN CURRENT_DATE - INTERVAL '30 days' AND CURRENT_DATE {limit_clause};"
                ]
                
                for query in date_queries:
                    examples.append({
                        'type': 'sql',
                        'content': query
                    })
        
        return examples

    def _generate_subquery_patterns(self, tables_info: Dict, db_type: str) -> List[Dict]:
        """Generate subquery patterns"""
        examples = []
        limit_clause = "TOP 10" if db_type == 'mssql' else "LIMIT 10"
        
        for table, info in tables_info.items():
            table_name = info['full_name']
            
            if info['numeric_cols']:
                for numeric_col in info['numeric_cols']:
                    subquery_patterns = [
                        f"SELECT * FROM {table_name} WHERE {numeric_col} > (SELECT AVG({numeric_col}) FROM {table_name}) {limit_clause};",
                        f"SELECT * FROM {table_name} WHERE {numeric_col} = (SELECT MAX({numeric_col}) FROM {table_name});",
                        f"SELECT * FROM {table_name} WHERE {numeric_col} = (SELECT MIN({numeric_col}) FROM {table_name});",
                        f"SELECT * FROM {table_name} WHERE {numeric_col} IN (SELECT {numeric_col} FROM {table_name} WHERE {numeric_col} > 0) {limit_clause};",
                        f"SELECT * FROM {table_name} t1 WHERE EXISTS (SELECT 1 FROM {table_name} t2 WHERE t2.{numeric_col} > t1.{numeric_col}) {limit_clause};",
                        f"SELECT * FROM {table_name} t1 WHERE NOT EXISTS (SELECT 1 FROM {table_name} t2 WHERE t2.{numeric_col} > t1.{numeric_col}) {limit_clause};"
                    ]
                    
                    for query in subquery_patterns:
                        examples.append({
                            'type': 'sql',
                            'content': query
                        })
        
        return examples

    def _generate_window_patterns(self, tables_info: Dict, db_type: str) -> List[Dict]:
        """Generate window function patterns"""
        if db_type in ['sqlite']:  # Skip for databases that don't support window functions well
            return []
            
        examples = []
        limit_clause = "TOP 10" if db_type == 'mssql' else "LIMIT 10"
        
        for table, info in tables_info.items():
            table_name = info['full_name']
            
            if info['numeric_cols']:
                for numeric_col in info['numeric_cols']:
                    window_queries = [
                        f"SELECT *, ROW_NUMBER() OVER (ORDER BY {numeric_col} DESC) as rank FROM {table_name} {limit_clause};",
                        f"SELECT *, RANK() OVER (ORDER BY {numeric_col} DESC) as rank FROM {table_name} {limit_clause};",
                        f"SELECT *, DENSE_RANK() OVER (ORDER BY {numeric_col} DESC) as dense_rank FROM {table_name} {limit_clause};",
                        f"SELECT *, LAG({numeric_col}) OVER (ORDER BY {numeric_col}) as previous_value FROM {table_name} {limit_clause};",
                        f"SELECT *, LEAD({numeric_col}) OVER (ORDER BY {numeric_col}) as next_value FROM {table_name} {limit_clause};"
                    ]
                    
                    if info['text_cols']:
                        group_col = info['text_cols'][0]
                        window_queries.extend([
                            f"SELECT *, ROW_NUMBER() OVER (PARTITION BY {group_col} ORDER BY {numeric_col} DESC) as rank_within_group FROM {table_name} {limit_clause};",
                            f"SELECT *, AVG({numeric_col}) OVER (PARTITION BY {group_col}) as group_average FROM {table_name} {limit_clause};",
                            f"SELECT *, SUM({numeric_col}) OVER (PARTITION BY {group_col}) as group_total FROM {table_name} {limit_clause};"
                        ])
                    
                    for query in window_queries:
                        examples.append({
                            'type': 'sql',
                            'content': query
                        })
        
        return examples

    def _generate_business_patterns(self, tables_info: Dict, db_type: str) -> List[Dict]:
        """Generate business logic patterns"""
        examples = []
        
        # Add comprehensive business documentation
        business_docs = [
            "Business users often need to analyze trends over time periods",
            "Revenue calculations should always exclude cancelled or refunded transactions",
            "Customer analysis requires grouping by various demographic segments",
            "Performance metrics are typically calculated as percentages and ratios",
            "Time-based analysis often uses rolling averages and period-over-period comparisons",
            "Data quality checks should verify for null values and outliers",
            "Reporting queries often need both summary and detailed breakdowns",
            "Financial reports require accurate date range filtering",
            "Marketing analysis focuses on conversion rates and customer acquisition costs",
            "Sales data analysis includes pipeline stages and win/loss tracking",
            "User behavior analysis tracks engagement and retention metrics",
            "Inventory management requires stock level monitoring and turnover rates",
            "Compliance reporting needs audit trails and data lineage",
            "Performance dashboards show key metrics with drill-down capabilities",
            "Operational reports track efficiency and resource utilization"
        ]
        
        for doc in business_docs:
            examples.append({
                'type': 'documentation',
                'content': doc
            })
        
        return examples

    def _sample_real_data(self, engine, tables_info: Dict, db_type: str) -> List[Dict]:
        """Sample real data for better context"""
        examples = []
        
        for table, info in tables_info.items():
            try:
                table_name = info['full_name']
                # Get a few sample rows
                query = f"SELECT * FROM {table_name} LIMIT 3"
                
                with engine.connect() as conn:
                    result = conn.execute(text(query))
                    rows = result.fetchall()
                    columns = list(result.keys())
                    
                if rows:
                    sample_data = f"Sample data from {table}:\n"
                    sample_data += f"Columns: {', '.join(columns)}\n"
                    for i, row in enumerate(rows):
                        sample_data += f"Row {i+1}: {dict(zip(columns, row))}\n"
                    
                    examples.append({
                        'type': 'documentation',
                        'content': sample_data
                    })
                    
                    # Generate context-aware queries based on real data
                    if rows:
                        first_row = dict(zip(columns, rows[0]))
                        for col_name, value in first_row.items():
                            if value is not None:
                                if isinstance(value, str):
                                    examples.append({
                                        'type': 'sql',
                                        'content': f"SELECT * FROM {table_name} WHERE {col_name} = '{value}' LIMIT 10;"
                                    })
                                elif isinstance(value, (int, float)):
                                    examples.append({
                                        'type': 'sql',
                                        'content': f"SELECT * FROM {table_name} WHERE {col_name} >= {value} LIMIT 10;"
                                    })
                                
            except Exception as e:
                print(f"⚠️  Could not sample data from {table}: {e}")
        
        return examples

    def _train_in_batches(self, examples: List[Dict], batch_size: int) -> Dict:
        """Train with all examples in batches"""
        stats = {
            'total_examples': len(examples),
            'ddl_count': len([e for e in examples if e['type'] == 'ddl']),
            'sql_count': len([e for e in examples if e['type'] == 'sql']),
            'doc_count': len([e for e in examples if e['type'] == 'documentation']),
            'batches_processed': 0,
            'errors': []
        }
        
        print(f"🔥 Training with {stats['total_examples']} examples:")
        print(f"   📋 DDL: {stats['ddl_count']}")
        print(f"   🔍 SQL: {stats['sql_count']}")  
        print(f"   📚 Documentation: {stats['doc_count']}")
        
        # Train in batches
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i+batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(examples) + batch_size - 1) // batch_size
            
            print(f"🔥 Processing batch {batch_num}/{total_batches} ({len(batch)} examples)")
            
            try:
                for example in batch:
                    if example['type'] == 'ddl':
                        self.train(ddl=example['content'])
                    elif example['type'] == 'sql':
                        self.train(sql=example['content'])
                    elif example['type'] == 'documentation':
                        self.train(documentation=example['content'])
                
                stats['batches_processed'] += 1
                print("✅", end="", flush=True)
                
            except Exception as e:
                error_msg = f"Batch {batch_num} error: {str(e)}"
                stats['errors'].append(error_msg)
                print(f"❌ {error_msg}")
        
        print(f"\n🎉 Completed {stats['batches_processed']}/{total_batches} batches")
        return stats

    def save_training_metadata(self, metadata: Dict[str, Any]):
        """Save training metadata for future reference"""
        metadata_file = self.model_path / "training_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        info = {
            "model_path": str(self.model_path),
            "is_trained": self.is_trained(),
            "config": {k: v for k, v in self.config.items() if k != 'api_key'}  # Don't expose API key
        }
        
        # Get training data count if available
        try:
            training_data = self.get_training_data()
            if training_data is not None:
                info["training_data_count"] = len(training_data)
            else:
                info["training_data_count"] = 0
        except:
            info["training_data_count"] = 0
            
        return info

# Factory function for creating beast mode trainers
def create_beast_mode_trainer(openai_api_key: str, model_path: str, model_name: str = "gpt-4o") -> BeastModeVannaTrainer:
    """
    Factory function to create a BEAST MODE Vanna trainer
    """
    config = {
        'api_key': openai_api_key,
        'model': model_name,
        'path': model_path
    }
    
    return BeastModeVannaTrainer(config=config)

# Utility functions
def generate_model_id(connection_params: Dict[str, Any]) -> str:
    """Generate a unique model ID based on connection parameters"""
    safe_params = {k: v for k, v in connection_params.items() if k != 'password'}
    param_string = json.dumps(safe_params, sort_keys=True)
    return hashlib.md5(param_string.encode()).hexdigest()

def get_model_storage_path(base_path: str, model_id: str) -> str:
    """Get the storage path for a specific model"""
    return str(Path(base_path) / f"model_{model_id}")

def load_trained_beast_model(openai_api_key: str, model_path: str, model_name: str = "gpt-4o") -> BeastModeVannaTrainer:
    """
    Load a previously trained BEAST MODE Vanna model for inference.
    """
    config = {
        'api_key': openai_api_key,
        'model': model_name,
        'path': model_path
    }
    return BeastModeVannaTrainer(config=config)