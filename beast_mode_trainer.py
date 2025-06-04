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
        
        # Discover business relationships
        relationships = self._discover_business_relationships(tables_info, engine)
        
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
        
        # 11. Dynamic business queries
        print("💼 Generating dynamic business queries...")
        business_queries = self._generate_dynamic_business_queries(tables_info, relationships, db_type)
        all_examples.extend(business_queries)
        print(f"   Generated {len(business_queries)} business logic examples")
        
        # 12. Natural language to SQL mappings
        print("🗣️ Generating natural language mappings...")
        nl_mappings = self._generate_natural_language_mappings(tables_info, relationships, db_type)
        all_examples.extend(nl_mappings)
        print(f"   Generated {len(nl_mappings)} natural language examples")
        
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
        """Extract comprehensive schema information with universal pattern discovery"""
        tables_info = {}
        
        if db_type == 'postgresql':
            schemas = [s for s in inspector.get_schema_names() if s not in ('pg_catalog', 'information_schema')]
            schema = schemas[0] if schemas else 'public'
        else:
            schema = None
        
        tables = inspector.get_table_names(schema=schema)
        print(f"🔍 Discovered {len(tables)} tables: {', '.join(tables)}")
        
        for table in tables:
            columns = inspector.get_columns(table, schema=schema)
            
            # Universal categorization by data types only
            numeric_cols = []
            text_cols = []
            date_cols = []
            boolean_cols = []
            
            for col in columns:
                col_type = str(col['type']).lower()
                col_name = col['name']
                
                # Pure data type categorization (no business assumptions)
                if any(t in col_type for t in ['int', 'float', 'decimal', 'numeric', 'money', 'double']):
                    numeric_cols.append(col_name)
                elif any(t in col_type for t in ['varchar', 'text', 'char', 'string']):
                    text_cols.append(col_name)
                elif any(t in col_type for t in ['date', 'time', 'timestamp']):
                    date_cols.append(col_name)
                elif any(t in col_type for t in ['bool', 'bit']):
                    boolean_cols.append(col_name)
            
            # Get relationships
            try:
                foreign_keys = inspector.get_foreign_keys(table, schema=schema)
                primary_keys = inspector.get_pk_constraint(table, schema=schema)
            except:
                foreign_keys = []
                primary_keys = {'constrained_columns': []}
            
            # Universal column analysis (no business assumptions)
            column_analysis = self._analyze_column_semantics(columns)
            
            tables_info[table] = {
                'schema': schema,
                'columns': columns,
                'numeric_cols': numeric_cols,
                'text_cols': text_cols,
                'date_cols': date_cols,
                'boolean_cols': boolean_cols,
                'foreign_keys': foreign_keys,
                'primary_keys': primary_keys.get('constrained_columns', []),
                'full_name': f"{schema}.{table}" if schema else table,
                'column_semantics': column_analysis  # Universal semantic analysis
            }
            
            print(f"   📋 {table}: {len(columns)} columns - {len(numeric_cols)} numeric, {len(text_cols)} text, {len(date_cols)} temporal")
        
        return tables_info

    def _analyze_column_semantics(self, columns: List[Dict]) -> Dict[str, str]:
        """Analyze column semantics universally without business assumptions"""
        semantics = {}
        
        for col in columns:
            col_name = col['name'].lower()
            col_type = str(col['type']).lower()
            
            # Universal semantic patterns (not business-specific)
            if col_name in ['id', 'pk', 'key'] or col_name.endswith('_id'):
                semantics[col['name']] = 'identifier'
            elif 'name' in col_name or 'title' in col_name:
                semantics[col['name']] = 'descriptive_text'
            elif any(term in col_name for term in ['code', 'abbreviation', 'abbr']):
                semantics[col['name']] = 'code_text'
            elif any(term in col_name for term in ['category', 'type', 'class', 'group']):
                semantics[col['name']] = 'categorical_text'
            elif any(term in col_name for term in ['status', 'state', 'flag']):
                semantics[col['name']] = 'status_indicator'
            elif any(term in col_name for term in ['description', 'note', 'comment']):
                semantics[col['name']] = 'free_text'
            elif any(term in col_name for term in ['email', 'phone', 'address']):
                semantics[col['name']] = 'contact_info'
            elif any(term in col_name for term in ['url', 'link', 'path']):
                semantics[col['name']] = 'reference_text'
            elif any(t in col_type for t in ['int', 'float', 'decimal', 'numeric']):
                if any(term in col_name for term in ['count', 'num', 'quantity', 'qty']):
                    semantics[col['name']] = 'count_metric'
                elif any(term in col_name for term in ['amount', 'value', 'price', 'cost', 'fee', 'sum', 'total']):
                    semantics[col['name']] = 'monetary_metric'
                elif any(term in col_name for term in ['rate', 'percent', 'ratio', 'score']):
                    semantics[col['name']] = 'calculated_metric'
                else:
                    semantics[col['name']] = 'general_metric'
            elif any(t in col_type for t in ['date', 'time', 'timestamp']):
                if any(term in col_name for term in ['created', 'start', 'begin']):
                    semantics[col['name']] = 'start_datetime'
                elif any(term in col_name for term in ['updated', 'modified', 'end', 'finish']):
                    semantics[col['name']] = 'end_datetime'
                else:
                    semantics[col['name']] = 'general_datetime'
            else:
                semantics[col['name']] = 'general_column'
        
        return semantics

    def _discover_business_relationships(self, tables_info: Dict, engine) -> Dict[str, Any]:
        """Dynamically discover universal data relationships without business assumptions"""
        print("🔍 Discovering universal data patterns...")
        
        relationships = {
            'entity_tables': [],  # Tables with potential entity data
            'metric_tables': [],  # Tables with numeric metrics
            'temporal_tables': [], # Tables with date/time data
            'primary_entity_column': None,
            'primary_metric_column': None,
            'cross_table_joins': [],
            'data_patterns': {}
        }
        
        # Universal pattern discovery - no business assumptions
        for table, info in tables_info.items():
            
            # Identify entity columns (text columns that likely represent entities)
            entity_cols = []
            for col_name in info['text_cols']:
                col_lower = col_name.lower()
                # Universal entity patterns (not business-specific)
                if (col_lower.endswith('_name') or col_lower.endswith('_id') or 
                    'name' in col_lower or 'code' in col_lower or 
                    'category' in col_lower or 'type' in col_lower or
                    'status' in col_lower or 'group' in col_lower):
                    entity_cols.append(col_name)
            
            # Identify metric columns (numeric columns for analysis)
            metric_cols = info['numeric_cols']
            
            # Identify temporal columns
            temporal_cols = info['date_cols']
            
            if entity_cols:
                relationships['entity_tables'].append({
                    'table': table,
                    'entity_columns': entity_cols,
                    'column_patterns': self._analyze_column_patterns(entity_cols)
                })
                
                # Set primary entity column (most common pattern)
                if not relationships['primary_entity_column']:
                    # Prefer columns with 'name' in them, then 'id', then first available
                    primary_col = next((col for col in entity_cols if 'name' in col.lower()), 
                                     next((col for col in entity_cols if 'id' in col.lower()), 
                                          entity_cols[0]))
                    relationships['primary_entity_column'] = primary_col
            
            if metric_cols:
                relationships['metric_tables'].append({
                    'table': table,
                    'metric_columns': metric_cols,
                    'metric_patterns': self._analyze_metric_patterns(metric_cols)
                })
                
                # Set primary metric column (prefer larger/common numeric values)
                if not relationships['primary_metric_column']:
                    relationships['primary_metric_column'] = metric_cols[0]
            
            if temporal_cols:
                relationships['temporal_tables'].append({
                    'table': table,
                    'temporal_columns': temporal_cols
                })
        
        # Discover cross-table relationships universally
        for table, info in tables_info.items():
            for other_table, other_info in tables_info.items():
                if table != other_table:
                    # Check for common column names (potential joins)
                    common_cols = set(col['name'] for col in info['columns']) & set(col['name'] for col in other_info['columns'])
                    if common_cols:
                        relationships['cross_table_joins'].append({
                            'table1': table,
                            'table2': other_table,
                            'common_columns': list(common_cols)
                        })
        
        # Sample data to understand universal patterns (no business assumptions)
        try:
            if relationships['entity_tables']:
                sample_table = relationships['entity_tables'][0]['table']
                entity_col = relationships['entity_tables'][0]['entity_columns'][0]
                
                with engine.connect() as conn:
                    query = text(f"SELECT DISTINCT {entity_col} FROM {sample_table} LIMIT 10")
                    result = conn.execute(query)
                    entities = [row[0] for row in result if row[0]]
                    relationships['data_patterns']['sample_entities'] = entities[:5]
                    print(f"   📊 Sample entities: {', '.join(str(e) for e in entities[:3])}")
        except Exception as e:
            print(f"   ⚠️  Could not sample entity data: {e}")
        
        print(f"   ✅ Found {len(relationships['entity_tables'])} entity tables")
        print(f"   ✅ Found {len(relationships['metric_tables'])} metric tables")
        print(f"   ✅ Found {len(relationships['temporal_tables'])} temporal tables")
        print(f"   ✅ Primary entity column: {relationships['primary_entity_column']}")
        print(f"   ✅ Primary metric column: {relationships['primary_metric_column']}")
        
        return relationships

    def _analyze_column_patterns(self, columns: List[str]) -> Dict[str, str]:
        """Analyze column patterns to understand data types universally"""
        patterns = {}
        for col in columns:
            col_lower = col.lower()
            if 'name' in col_lower:
                patterns[col] = 'descriptive_entity'
            elif 'id' in col_lower:
                patterns[col] = 'identifier_entity'
            elif 'code' in col_lower:
                patterns[col] = 'code_entity'
            elif any(term in col_lower for term in ['category', 'type', 'group', 'class']):
                patterns[col] = 'categorical_entity'
            elif any(term in col_lower for term in ['status', 'state', 'flag']):
                patterns[col] = 'status_entity'
            else:
                patterns[col] = 'general_entity'
        return patterns

    def _analyze_metric_patterns(self, columns: List[str]) -> Dict[str, str]:
        """Analyze metric patterns universally"""
        patterns = {}
        for col in columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['count', 'num', 'number']):
                patterns[col] = 'count_metric'
            elif any(term in col_lower for term in ['amount', 'value', 'price', 'cost', 'fee']):
                patterns[col] = 'monetary_metric'
            elif any(term in col_lower for term in ['rate', 'percent', 'ratio']):
                patterns[col] = 'ratio_metric'
            elif any(term in col_lower for term in ['score', 'rating', 'rank']):
                patterns[col] = 'score_metric'
            elif any(term in col_lower for term in ['size', 'length', 'weight', 'volume']):
                patterns[col] = 'measurement_metric'
            else:
                patterns[col] = 'general_metric'
        return patterns

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
        """Generate universal data patterns based on discovered schema - no business assumptions"""
        examples = []
        
        # Universal pattern discovery (works for any domain)
        has_entity_data = any(info['text_cols'] for info in tables_info.values())
        has_numeric_data = any(info['numeric_cols'] for info in tables_info.values())
        has_temporal_data = any(info['date_cols'] for info in tables_info.values())
        has_categorical_data = any(len(info['text_cols']) > 1 for info in tables_info.values())
        
        # Generate universal documentation based on discovered patterns
        universal_docs = [
            f"This database contains {len(tables_info)} tables for data analysis",
            "Users can analyze data patterns, trends, and relationships across tables",
        ]
        
        if has_entity_data:
            universal_docs.extend([
                "Text columns can be used for grouping, filtering, and categorizing data",
                "Entity analysis involves identifying distinct values and their frequencies",
                "Text-based filtering supports pattern matching and exact value searches",
                "Categorical analysis groups related items for comparison"
            ])
        
        if has_numeric_data:
            universal_docs.extend([
                "Numeric columns support mathematical operations: SUM, AVG, MIN, MAX, COUNT",
                "Aggregation analysis combines numeric values across groups",
                "Statistical analysis calculates averages, totals, and distributions",
                "Numeric comparisons use operators: >, <, =, BETWEEN, IN"
            ])
        
        if has_temporal_data:
            universal_docs.extend([
                "Date/time columns enable temporal analysis and trending",
                "Time-based analysis uses date functions for grouping by periods",
                "Temporal filtering supports date ranges and specific time periods",
                "Trend analysis compares data across different time intervals"
            ])
        
        if has_categorical_data:
            universal_docs.extend([
                "Multiple text columns enable multi-dimensional analysis",
                "Cross-tabulation shows relationships between categorical variables",
                "Hierarchical grouping supports drill-down analysis",
                "Category combinations reveal data distribution patterns"
            ])
        
        # Add table-specific universal context
        for table, info in tables_info.items():
            table_purpose = self._infer_universal_table_purpose(table, info)
            universal_docs.append(f"Table {table} serves as: {table_purpose}")
            
            # Universal column documentation
            if info['numeric_cols'] and info['text_cols']:
                universal_docs.append(f"Table {table} supports entity-metric analysis using text columns for grouping and numeric columns for calculations")
            
            if info['date_cols']:
                universal_docs.append(f"Table {table} enables temporal analysis with date/time tracking")
            
            # Semantic documentation based on column analysis
            if 'column_semantics' in info:
                semantic_summary = {}
                for col, semantic in info['column_semantics'].items():
                    semantic_summary[semantic] = semantic_summary.get(semantic, 0) + 1
                
                for semantic_type, count in semantic_summary.items():
                    if count > 1:
                        universal_docs.append(f"Table {table} has {count} columns of type '{semantic_type}' for related analysis")
        
        # Add universal query pattern documentation
        universal_docs.extend([
            "For ranking analysis, use ORDER BY with DESC/ASC and LIMIT clauses",
            "For aggregation analysis, use GROUP BY with SUM, COUNT, AVG functions",
            "For filtering analysis, use WHERE clauses with appropriate operators",
            "For temporal analysis, use date functions and time-based grouping",
            "For relationship analysis, use JOIN operations between related tables",
            "For statistical analysis, use aggregation functions with HAVING clauses",
            "Always handle NULL values appropriately in calculations and comparisons",
            "Use meaningful aliases for calculated columns and aggregated results"
        ])
        
        for doc in universal_docs:
            examples.append({
                'type': 'documentation',
                'content': doc
            })
        
        return examples
    
    def _infer_universal_table_purpose(self, table_name: str, table_info: Dict) -> str:
        """Infer universal table purpose based on structure, not business domain"""
        
        # Analyze table structure patterns
        num_columns = len(table_info['columns'])
        num_numeric = len(table_info['numeric_cols'])
        num_text = len(table_info['text_cols'])
        num_dates = len(table_info['date_cols'])
        has_ids = any(col.lower().endswith('_id') or col.lower() == 'id' for col in [c['name'] for c in table_info['columns']])
        
        # Universal patterns based on column composition
        if num_dates > 2 and num_numeric > 3:
            return "time-series data with multiple metrics for temporal analysis"
        elif has_ids and num_text > num_numeric:
            return "reference/master data for entity management"
        elif num_numeric > num_text and num_numeric > 3:
            return "metrics/measurements data for analytical calculations"
        elif num_text > num_numeric and num_text > 3:
            return "descriptive/categorical data for classification and grouping"
        elif table_info['foreign_keys']:
            return "relational data linking to other entities"
        elif num_columns < 5:
            return "simple lookup/reference data"
        elif num_dates > 0 and num_numeric > 0:
            return "transactional/event data with temporal tracking"
        else:
            return "general purpose data storage for mixed analysis"

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
                    elif example['type'] == 'question_sql':
                        self.train(question=example['question'], sql=example['sql'])
                
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

    def ask(self, question: str, **kwargs) -> str:
        """
        Simplified ask method that uses our robust generate_sql
        """
        try:
            # Use our custom generate_sql method directly
            sql = self.generate_sql(question, **kwargs)
            
            if sql and sql.strip():
                print(f"✅ ask() returning SQL: {sql}")
                return sql
            else:
                print(f"❌ ask() - no SQL generated")
                return None
                
        except Exception as e:
            print(f"❌ Error in ask method: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_sql(self, question: str, **kwargs) -> str:
        """
        Override generate_sql to properly handle modern LLM responses
        This method is called by ask() and should return clean SQL
        """
        try:
            # Get the full prompt using parent's method
            question_sql_list = self.get_similar_question_sql(question)
            ddl_list = self.get_related_ddl(question)
            doc_list = self.get_related_documentation(question)
            
            # Create a much more explicit initial prompt for SQL generation
            initial_prompt = f"""You are a SQL expert. Generate ONLY valid SQL code to answer this question: {question}

IMPORTANT INSTRUCTIONS:
- Return ONLY executable SQL code, no explanations or descriptions
- Do NOT list tables or describe data - only provide runnable SQL
- Use proper SQL syntax with SELECT, FROM, WHERE, etc.
- End your SQL with a semicolon (;)
- If you need to show table information, use SQL queries like "SELECT * FROM table_name LIMIT 5;"

Question: {question}
SQL Query:"""
            
            prompt = self.get_sql_prompt(
                question=question,
                question_sql_list=question_sql_list,
                ddl_list=ddl_list,
                doc_list=doc_list,
                initial_prompt=initial_prompt
            )
            
            # Submit to LLM
            llm_response = self.submit_prompt(prompt, **kwargs)
            
            print(f"🤖 LLM Response: {llm_response}")
            
            # Modern extraction - trust the LLM's structure
            if not llm_response:
                print("❌ No response from LLM")
                return None
            
            # Clean the response
            import re
            
            # 1. Handle markdown SQL blocks (most common)
            if '```sql' in llm_response.lower():
                match = re.search(r'```sql\s*\n(.*?)\n```', llm_response, re.DOTALL | re.IGNORECASE)
                if match:
                    sql = match.group(1).strip()
                    print(f"✅ Extracted SQL from markdown: {sql}")
                    return sql
            
            # 2. Handle generic markdown blocks
            if '```' in llm_response:
                match = re.search(r'```\s*\n(.*?)\n```', llm_response, re.DOTALL)
                if match:
                    sql = match.group(1).strip()
                    # Verify it's SQL
                    if any(kw in sql.upper() for kw in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']):
                        print(f"✅ Extracted SQL from code block: {sql}")
                        return sql
            
            # 3. Direct SQL (when LLM returns just SQL)
            lines = llm_response.strip().split('\n')
            if lines and any(lines[0].upper().startswith(kw) for kw in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER', 'WITH']):
                # It's likely pure SQL, return as is
                sql = llm_response.strip()
                print(f"✅ Direct SQL response: {sql}")
                return sql
            
            # 4. Extract SQL from response - Enhanced pattern matching
            sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER', 'WITH']
            for i, line in enumerate(lines):
                if any(line.strip().upper().startswith(kw) for kw in sql_keywords):
                    # Found SQL start, collect until end
                    sql_lines = []
                    for j in range(i, len(lines)):
                        line = lines[j].strip()
                        if line and line != '```':
                            sql_lines.append(line)
                            if line.endswith(';'):
                                break
                    if sql_lines:
                        sql = '\n'.join(sql_lines)
                        print(f"✅ Extracted SQL from line {i}: {sql}")
                        return sql
            
            # 5. Try to find any SQL-like pattern in the response
            sql_pattern = re.search(r'(SELECT.*?;)', llm_response, re.DOTALL | re.IGNORECASE)
            if sql_pattern:
                sql = sql_pattern.group(1).strip()
                print(f"✅ Extracted SQL with pattern matching: {sql}")
                return sql
            
            # 6. Last resort - if response looks like it might contain SQL, try to guide the LLM
            if any(word.upper() in llm_response.upper() for word in ['TABLE', 'COLUMN', 'DATABASE', 'ROWS']):
                print(f"⚠️ Response contains database terms but no SQL. Attempting to re-prompt...")
                
                # Try a more direct approach
                direct_prompt = f"""Convert this to executable SQL code: {question}
                
Available tables: {', '.join([ddl.split()[2] for ddl in ddl_list if 'CREATE TABLE' in ddl][:5])}

Return only the SQL query, nothing else:"""
                
                retry_response = self.submit_prompt(direct_prompt, **kwargs)
                print(f"🔄 Retry response: {retry_response}")
                
                # Try to extract from retry
                if retry_response and any(kw in retry_response.upper() for kw in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
                    sql_pattern = re.search(r'(SELECT.*?;)', retry_response, re.DOTALL | re.IGNORECASE)
                    if sql_pattern:
                        sql = sql_pattern.group(1).strip()
                        print(f"✅ Extracted SQL from retry: {sql}")
                        return sql
            
            # If no SQL found, return None instead of the raw response
            print(f"⚠️ No SQL found in response: {llm_response}")
            return None
            
        except Exception as e:
            print(f"❌ Error in generate_sql method: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_dynamic_business_queries(self, tables_info: Dict, relationships: Dict, db_type: str) -> List[Dict]:
        """Generate universal queries based on discovered relationships (renamed for compatibility)"""
        examples = []
        limit_clause = "TOP 10" if db_type == 'mssql' else "LIMIT 10"
        
        # Generate entity analysis queries
        if relationships['entity_tables'] and relationships['primary_entity_column']:
            entity_col = relationships['primary_entity_column']
            
            for table_info in relationships['entity_tables']:
                table = table_info['table']
                table_full = tables_info[table]['full_name']
                
                # Basic entity queries
                entity_queries = [
                    f"-- Find all unique entities\nSELECT DISTINCT {entity_col} FROM {table_full} ORDER BY {entity_col} {limit_clause};",
                    f"-- Count records per entity\nSELECT {entity_col}, COUNT(*) as record_count FROM {table_full} GROUP BY {entity_col} ORDER BY record_count DESC {limit_clause};",
                    f"-- Find top entities by activity\nSELECT {entity_col}, COUNT(*) as activity_count FROM {table_full} WHERE {entity_col} IS NOT NULL GROUP BY {entity_col} ORDER BY activity_count DESC {limit_clause};"
                ]
                
                # Metric analysis if metric columns exist
                if table in [t['table'] for t in relationships['metric_tables']]:
                    metric_cols = next(t['metric_columns'] for t in relationships['metric_tables'] if t['table'] == table)
                    for metric_col in metric_cols[:2]:  # Top 2 metrics
                        entity_queries.extend([
                            f"-- Top entities by {metric_col}\nSELECT {entity_col}, SUM({metric_col}) as total_{metric_col} FROM {table_full} WHERE {entity_col} IS NOT NULL GROUP BY {entity_col} ORDER BY total_{metric_col} DESC {limit_clause};",
                            f"-- Average {metric_col} per entity\nSELECT {entity_col}, AVG({metric_col}) as avg_{metric_col}, COUNT(*) as transactions FROM {table_full} WHERE {entity_col} IS NOT NULL GROUP BY {entity_col} HAVING COUNT(*) > 1 ORDER BY avg_{metric_col} DESC {limit_clause};"
                        ])
                
                for query in entity_queries:
                    examples.append({'type': 'sql', 'content': query})
        
        # Generate cross-table analysis queries
        for join_info in relationships['cross_table_joins']:
            table1 = join_info['table1']
            table2 = join_info['table2']
            common_cols = join_info['common_columns']
            
            if common_cols and relationships['primary_entity_column'] in common_cols:
                join_col = relationships['primary_entity_column']
                table1_full = tables_info[table1]['full_name']
                table2_full = tables_info[table2]['full_name']
                
                cross_queries = [
                    f"-- Cross-table entity analysis\nSELECT t1.{join_col}, COUNT(DISTINCT t1.*) as {table1}_records, COUNT(DISTINCT t2.*) as {table2}_records FROM {table1_full} t1 LEFT JOIN {table2_full} t2 ON t1.{join_col} = t2.{join_col} GROUP BY t1.{join_col} ORDER BY {table1}_records DESC {limit_clause};"
                ]
                
                for query in cross_queries:
                    examples.append({'type': 'sql', 'content': query})
        
        # Generate universal "top entity" query
        if relationships['entity_tables'] and relationships['primary_entity_column']:
            entity_col = relationships['primary_entity_column']
            
            # Build dynamic UNION query for all entity tables
            union_parts = []
            for table_info in relationships['entity_tables']:
                table = table_info['table']
                table_full = tables_info[table]['full_name']
                union_parts.append(f"SELECT '{table}' as source_table, {entity_col}, COUNT(*) as count FROM {table_full} WHERE {entity_col} IS NOT NULL GROUP BY {entity_col}")
            
            if union_parts:
                universal_query = f"""-- Find the top entity across all tables
WITH entity_counts AS (
    {' UNION ALL '.join(union_parts)}
)
SELECT 
    {entity_col},
    SUM(count) as total_records,
    COUNT(DISTINCT source_table) as tables_present_in,
    STRING_AGG(DISTINCT source_table, ', ') as found_in_tables
FROM entity_counts
GROUP BY {entity_col}
ORDER BY total_records DESC
LIMIT 1;"""
                
                examples.append({'type': 'sql', 'content': universal_query})
        
        return examples

    def _generate_natural_language_mappings(self, tables_info: Dict, relationships: Dict, db_type: str) -> List[Dict]:
        """Generate universal natural language to SQL mappings based on discovered patterns"""
        examples = []
        
        if not relationships['entity_tables']:
            return examples
        
        entity_col = relationships['primary_entity_column']
        metric_col = relationships['primary_metric_column']
        
        # Generate universal question-SQL pairs for any domain
        nl_sql_pairs = []
        
        # Universal "top entity" questions (works for any domain)
        if entity_col:
            # Build dynamic UNION query for all entity tables
            union_parts = []
            for table_info in relationships['entity_tables']:
                table = table_info['table']
                table_full = tables_info[table]['full_name']
                union_parts.append(f"SELECT '{table}' as source_table, {entity_col}, COUNT(*) as count FROM {table_full} WHERE {entity_col} IS NOT NULL GROUP BY {entity_col}")
            
            if union_parts:
                top_entity_sql = f"""WITH entity_counts AS (
    {' UNION ALL '.join(union_parts)}
)
SELECT 
    {entity_col},
    SUM(count) as total_records,
    COUNT(DISTINCT source_table) as tables_present_in,
    STRING_AGG(DISTINCT source_table, ', ') as found_in_tables
FROM entity_counts
GROUP BY {entity_col}
ORDER BY total_records DESC
LIMIT 1;"""
                
                # Universal ways to ask for top entity (no business assumptions)
                top_entity_questions = [
                    f"What is the top {entity_col}?",
                    f"Which {entity_col} has the most records?",
                    f"Show me the {entity_col} with highest activity",
                    f"Find the most common {entity_col}",
                    f"What {entity_col} appears most frequently?",
                    f"Which {entity_col} has the most data?",
                    "Show me the top entity",
                    "What entity has the most records?",
                    "Find the most active entity"
                ]
                
                for question in top_entity_questions:
                    nl_sql_pairs.append({
                        'question': question,
                        'sql': top_entity_sql
                    })
        
        # Universal metric-based questions (works for any numeric data)
        if entity_col and metric_col:
            for table_info in relationships['metric_tables']:
                table = table_info['table']
                table_full = tables_info[table]['full_name']
                
                # Check if this table also has entity columns
                table_entities = [col for col in tables_info[table]['text_cols'] 
                                if any(col == entity_info['entity_columns'][0] 
                                      for entity_info in relationships['entity_tables'] 
                                      if entity_info['table'] == table)]
                
                if table_entities:
                    entity_in_table = table_entities[0]
                    
                    universal_questions = [
                        f"Which {entity_in_table} has the highest {metric_col}?",
                        f"Show me top {entity_in_table} by {metric_col}",
                        f"What {entity_in_table} has the most {metric_col}?",
                        f"Rank {entity_in_table} by {metric_col}",
                        f"Find the best {entity_in_table} by {metric_col}",
                        "Show me the top performer",
                        "Which entity has the highest value?",
                        "Find the entity with maximum metric"
                    ]
                    
                    metric_sql = f"SELECT {entity_in_table}, SUM({metric_col}) as total_{metric_col} FROM {table_full} WHERE {entity_in_table} IS NOT NULL GROUP BY {entity_in_table} ORDER BY total_{metric_col} DESC LIMIT 1;"
                    
                    for question in universal_questions:
                        nl_sql_pairs.append({
                            'question': question,
                            'sql': metric_sql
                        })
        
        # Universal aggregation questions for any table
        for table, info in tables_info.items():
            if info['numeric_cols']:
                table_full = info['full_name']
                
                for metric_col in info['numeric_cols'][:2]:  # Top 2 numeric columns
                    universal_questions = [
                        f"What is the average {metric_col}?",
                        f"Show me the total {metric_col}",
                        f"What is the sum of {metric_col}?",
                        f"Calculate average {metric_col}",
                        f"Find total {metric_col}",
                        f"What is the maximum {metric_col}?",
                        f"What is the minimum {metric_col}?",
                        "Show me the average value",
                        "Calculate the total",
                        "Find the sum"
                    ]
                    
                    for question in universal_questions:
                        if 'average' in question.lower():
                            sql = f"SELECT AVG({metric_col}) as avg_{metric_col} FROM {table_full};"
                        elif any(word in question.lower() for word in ['total', 'sum']):
                            sql = f"SELECT SUM({metric_col}) as total_{metric_col} FROM {table_full};"
                        elif 'maximum' in question.lower() or 'max' in question.lower():
                            sql = f"SELECT MAX({metric_col}) as max_{metric_col} FROM {table_full};"
                        elif 'minimum' in question.lower() or 'min' in question.lower():
                            sql = f"SELECT MIN({metric_col}) as min_{metric_col} FROM {table_full};"
                        else:
                            sql = f"SELECT AVG({metric_col}) as avg_{metric_col} FROM {table_full};"
                        
                        nl_sql_pairs.append({
                            'question': question,
                            'sql': sql
                        })
        
        # Universal count questions
        for table, info in tables_info.items():
            table_full = info['full_name']
            
            count_questions = [
                f"How many records are in {table}?",
                f"Count records in {table}",
                f"How many rows in {table}?",
                f"What is the total count for {table}?",
                "Show me the record count",
                "How many entries are there?",
                "Count all records"
            ]
            
            count_sql = f"SELECT COUNT(*) as total_records FROM {table_full};"
            
            for question in count_questions:
                nl_sql_pairs.append({
                    'question': question,
                    'sql': count_sql
                })
        
        # Universal temporal questions (if date columns exist)
        for table, info in tables_info.items():
            if info['date_cols']:
                table_full = info['full_name']
                date_col = info['date_cols'][0]
                
                temporal_questions = [
                    f"Show me recent {table} data",
                    f"What are the latest records in {table}?",
                    f"Find recent entries in {table}",
                    f"Show me data from last 30 days",
                    "Show me recent records",
                    "Find latest entries",
                    "What are the newest records?"
                ]
                
                temporal_sql = f"SELECT * FROM {table_full} WHERE {date_col} >= CURRENT_DATE - INTERVAL '30 days' ORDER BY {date_col} DESC LIMIT 10;"
                
                for question in temporal_questions:
                    nl_sql_pairs.append({
                        'question': question,
                        'sql': temporal_sql
                    })
        
        # Convert to training examples
        for pair in nl_sql_pairs:
            examples.append({
                'type': 'question_sql',
                'question': pair['question'],
                'sql': pair['sql']
            })
        
        return examples

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