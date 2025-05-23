#!/usr/bin/env python3
"""
BEAST MODE Vanna.AI Training System - ENHANCED VERSION
Generates HUNDREDS of high-quality training examples with dynamic business context
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
    ENHANCED BEAST MODE Vanna implementation - generates 500+ training examples
    with dynamic business context awareness
    """
    
    def __init__(self, config=None):
        # Ensure we have a path for persistent storage
        if config and 'path' not in config:
            config['path'] = str(Path.cwd() / "vanna_storage" / "default_model")
        elif not config:
            config = {'path': str(Path.cwd() / "vanna_storage" / "default_model")}
            
        # Create the directory if it doesn't exist
        Path(config['path']).mkdir(parents=True, exist_ok=True)
        
        # CRITICAL FIX: Add collection name for ChromaDB
        if 'collection_name' not in config:
            # Generate a unique collection name based on the model path
            path_hash = hashlib.md5(str(config['path']).encode()).hexdigest()[:8]
            config['collection_name'] = f"vanna_beast_mode_{path_hash}"
        
        # Initialize parent classes with config
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)
        
        # Store config for later use
        self.config = config
        self.model_path = Path(config['path'])
        
        print(f"🔥 ENHANCED BEAST MODE Vanna trainer initialized")
        print(f"📁 Model path: {self.model_path}")
        print(f"🗃️  ChromaDB collection: {config['collection_name']}")

    def is_trained(self) -> bool:
        """Check if the model has been trained by looking for ChromaDB files"""
        chroma_db_file = self.model_path / "chroma.sqlite3"
        return chroma_db_file.exists()

    def beast_mode_training(self, engine, db_type: str, batch_size: int = 50) -> Dict[str, Any]:
        """
        ENHANCED BEAST MODE training - generates 500+ examples with dynamic business context
        """
        print("🔥 STARTING ENHANCED BEAST MODE TRAINING 🔥")
        print("🎯 Target: Generate 500+ high-quality training examples with dynamic business context")
        
        # Extract comprehensive database information
        inspector = inspect(engine)
        tables_info = self._extract_comprehensive_schema(engine, db_type, inspector)
        
        # DYNAMIC: Analyze database structure to understand business context
        business_context = self._analyze_business_context(tables_info, engine)
        
        # Generate MASSIVE amounts of training data
        all_examples = []
        
        # 0. DYNAMIC Business Context (generated from actual data)
        print("🧠 Generating dynamic business context...")
        context_examples = self._generate_dynamic_business_context(business_context, tables_info)
        all_examples.extend(context_examples)
        print(f"   Generated {len(context_examples)} business context examples")
        
        # 1. Enhanced DDL with detailed comments
        print("📋 Generating enhanced DDL statements...")
        ddl_examples = self._generate_enhanced_ddl(tables_info, db_type, business_context)
        all_examples.extend(ddl_examples)
        print(f"   Generated {len(ddl_examples)} DDL examples")
        
        # 2. Basic query patterns (simple SELECTs)
        print("🔍 Generating basic query patterns...")
        basic_queries = self._generate_basic_queries(tables_info, db_type, business_context)
        all_examples.extend(basic_queries)
        print(f"   Generated {len(basic_queries)} basic query examples")
        
        # 3. Advanced JOIN patterns
        print("🔗 Generating JOIN patterns...")
        join_queries = self._generate_join_patterns(tables_info, db_type, inspector)
        all_examples.extend(join_queries)
        print(f"   Generated {len(join_queries)} JOIN examples")
        
        # 4. Aggregation and GROUP BY patterns
        print("📊 Generating aggregation patterns...")
        agg_queries = self._generate_aggregation_patterns(tables_info, db_type, business_context)
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
        
        # 8. DYNAMIC Business logic patterns (based on actual data)
        print("💼 Generating dynamic business logic patterns...")
        business_patterns = self._generate_dynamic_business_patterns(tables_info, db_type, business_context)
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
        
        print(f"🔥 ENHANCED BEAST MODE: Generated {len(all_examples)} training examples!")
        
        # Train in batches with enhanced validation
        stats = self._train_in_batches(all_examples, batch_size)
        
        # Save comprehensive metadata
        metadata = {
            "training_mode": "ENHANCED_BEAST_MODE",
            "total_examples": len(all_examples),
            "successful_examples": stats.get('successful_examples', 0),
            "db_type": db_type,
            "tables_count": len(tables_info),
            "business_context": business_context,
            "training_stats": stats,
            "trained_at": str(pd.Timestamp.now())
        }
        self.save_training_metadata(metadata)
        
        print("🎉 ENHANCED BEAST MODE TRAINING COMPLETE!")
        print(f"📊 Total examples: {len(all_examples)}")
        print(f"✅ Successfully trained: {stats.get('successful_examples', 0)}")
        print(f"🎯 Your model is now a TRUE ENHANCED BEAST!")
        
        return stats

    def _analyze_business_context(self, tables_info: Dict, engine) -> Dict[str, Any]:
        """
        DYNAMIC: Analyze the database to understand business context
        """
        print("🔍 Analyzing database for business context...")
        
        context = {
            "entity_mappings": {},
            "metric_columns": [],
            "dimension_columns": [],
            "business_rules": [],
            "table_purposes": {},
            "common_patterns": []
        }
        
        # Analyze column names to understand business entities
        all_columns = []
        for table, info in tables_info.items():
            for col in info['columns']:
                all_columns.append({
                    'table': table,
                    'column': col['name'],
                    'type': str(col['type']).lower()
                })
        
        # Detect entity mappings dynamically
        customer_like_cols = [col for col in all_columns if any(term in col['column'].lower() for term in ['client', 'customer', 'user', 'account', 'brand'])]
        if customer_like_cols:
            primary_customer_col = max(customer_like_cols, key=lambda x: sum(term in x['column'].lower() for term in ['client', 'customer']))['column']
            context["entity_mappings"]["customers"] = primary_customer_col
            context["business_rules"].append(f"When users ask about 'customers', they mean '{primary_customer_col}' in the database")
        
        # Detect revenue/sales columns
        revenue_like_cols = [col for col in all_columns if any(term in col['column'].lower() for term in ['revenue', 'sales', 'amount', 'value', 'spend', 'cost'])]
        if revenue_like_cols:
            primary_revenue_col = max(revenue_like_cols, key=lambda x: 'revenue' in x['column'].lower())['column']
            context["entity_mappings"]["sales"] = primary_revenue_col
            context["business_rules"].append(f"When users ask about 'sales', they mean '{primary_revenue_col}' columns")
        
        # Detect metric columns (numeric)
        context["metric_columns"] = [col['column'] for col in all_columns if any(t in col['type'] for t in ['int', 'float', 'decimal', 'numeric'])]
        
        # Detect dimension columns (text)
        context["dimension_columns"] = [col['column'] for col in all_columns if any(t in col['type'] for t in ['varchar', 'text', 'char'])]
        
        # Analyze table purposes based on names and columns
        for table, info in tables_info.items():
            col_names = [col['name'].lower() for col in info['columns']]
            
            if any(term in table.lower() for term in ['campaign', 'ad', 'marketing']):
                context["table_purposes"][table] = "advertising_campaign_data"
                context["business_rules"].append(f"Table {table} contains advertising campaign performance data")
            elif any(term in col_names for term in ['revenue', 'sales', 'performance']):
                context["table_purposes"][table] = "performance_metrics"
                context["business_rules"].append(f"Table {table} contains performance metrics and revenue data")
            elif any(term in col_names for term in ['client', 'customer', 'brand']):
                context["table_purposes"][table] = "entity_master_data"
                context["business_rules"].append(f"Table {table} contains entity/customer information")
        
        # Sample actual data to understand values
        try:
            sample_table = list(tables_info.keys())[0]
            with engine.connect() as conn:
                sample_query = f"SELECT * FROM {sample_table} LIMIT 5"
                result = conn.execute(text(sample_query))
                rows = result.fetchall()
                if rows:
                    columns = list(result.keys())
                    sample_data = [dict(zip(columns, row)) for row in rows]
                    
                    # Analyze sample data for business patterns
                    for col in columns:
                        values = [row.get(col) for row in sample_data if row.get(col) is not None]
                        if values and isinstance(values[0], str):
                            context["common_patterns"].append(f"Column {col} contains values like: {', '.join(map(str, values[:3]))}")
                        
        except Exception as e:
            print(f"⚠️  Could not sample data for business context: {e}")
        
        print(f"✅ Business context analysis complete:")
        print(f"   🎯 Entity mappings: {context['entity_mappings']}")
        print(f"   📊 Metric columns: {len(context['metric_columns'])}")
        print(f"   📋 Dimension columns: {len(context['dimension_columns'])}")
        print(f"   📜 Business rules: {len(context['business_rules'])}")
        
        return context

    def _generate_dynamic_business_context(self, business_context: Dict, tables_info: Dict) -> List[Dict]:
        """
        Generate dynamic business context based on actual database analysis
        """
        examples = []
        
        # Generate business rules based on analysis
        for rule in business_context["business_rules"]:
            examples.append({
                'type': 'documentation',
                'content': rule
            })
        
        # Generate entity mapping documentation
        for logical_name, physical_col in business_context["entity_mappings"].items():
            examples.append({
                'type': 'documentation', 
                'content': f"Business Entity Mapping: '{logical_name}' maps to column '{physical_col}'"
            })
        
        # Generate table purpose documentation
        for table, purpose in business_context["table_purposes"].items():
            examples.append({
                'type': 'documentation',
                'content': f"Table {table} purpose: {purpose.replace('_', ' ')}"
            })
        
        # Generate metric and dimension documentation
        if business_context["metric_columns"]:
            examples.append({
                'type': 'documentation',
                'content': f"Key numeric metrics available: {', '.join(business_context['metric_columns'][:10])}"
            })
        
        if business_context["dimension_columns"]:
            examples.append({
                'type': 'documentation',
                'content': f"Key dimensions for grouping: {', '.join(business_context['dimension_columns'][:10])}"
            })
        
        # Generate specific business query patterns based on detected entities
        customer_col = business_context["entity_mappings"].get("customers")
        revenue_col = business_context["entity_mappings"].get("sales")
        
        if customer_col and revenue_col:
            main_table = list(tables_info.keys())[0]  # Use first table as primary
            
            # Critical question-SQL mappings
            critical_mappings = [
                {
                    "question": "Show me the top 10 customers by revenue",
                    "sql": f"SELECT {customer_col}, SUM({revenue_col}) as total_revenue FROM {main_table} GROUP BY {customer_col} ORDER BY total_revenue DESC LIMIT 10"
                },
                {
                    "question": "Top customers by sales",
                    "sql": f"SELECT {customer_col}, SUM({revenue_col}) as total_sales FROM {main_table} GROUP BY {customer_col} ORDER BY total_sales DESC LIMIT 10"
                },
                {
                    "question": "Which customers generated the most revenue",
                    "sql": f"SELECT {customer_col}, SUM({revenue_col}) as total_revenue FROM {main_table} GROUP BY {customer_col} ORDER BY total_revenue DESC"
                },
                {
                    "question": "Best customers by total sales volume", 
                    "sql": f"SELECT {customer_col}, SUM({revenue_col}) as sales_volume FROM {main_table} GROUP BY {customer_col} ORDER BY sales_volume DESC LIMIT 10"
                },
                {
                    "question": "Customer revenue ranking",
                    "sql": f"SELECT {customer_col}, SUM({revenue_col}) as revenue, RANK() OVER (ORDER BY SUM({revenue_col}) DESC) as rank FROM {main_table} GROUP BY {customer_col}"
                }
            ]
            
            for mapping in critical_mappings:
                examples.append({
                    'type': 'question_sql',
                    'question': mapping['question'],
                    'sql': mapping['sql']
                })
        
        return examples

    def _generate_enhanced_ddl(self, tables_info: Dict, db_type: str, business_context: Dict) -> List[Dict]:
        """Generate enhanced DDL with business context comments"""
        examples = []
        
        for table, info in tables_info.items():
            # Basic DDL
            columns_ddl = []
            comments = []
            
            # Add table purpose comment
            table_purpose = business_context["table_purposes"].get(table, "data storage")
            comments.append(f"-- Table {table}: {table_purpose}")
            
            for col in info['columns']:
                col_type = str(col['type'])
                nullable = "" if col.get('nullable', True) else " NOT NULL"
                column_def = f"{col['name']} {col_type}{nullable}"
                columns_ddl.append(column_def)
                
                # Add business context comments
                col_name = col['name']
                if col_name in business_context["metric_columns"]:
                    comments.append(f"-- {col_name}: Numeric metric for calculations and KPI analysis")
                elif col_name in business_context["dimension_columns"]:
                    comments.append(f"-- {col_name}: Dimension for grouping and filtering")
                elif col_name in business_context["entity_mappings"].values():
                    entity_type = [k for k, v in business_context["entity_mappings"].items() if v == col_name][0]
                    comments.append(f"-- {col_name}: Primary {entity_type} identifier")
            
            ddl = f"CREATE TABLE {table} (\n  " + ",\n  ".join(columns_ddl) + "\n);"
            
            examples.append({
                'type': 'ddl',
                'content': ddl + "\n" + "\n".join(comments)
            })
        
        return examples

    def _generate_basic_queries(self, tables_info: Dict, db_type: str, business_context: Dict) -> List[Dict]:
        """Generate basic query patterns with business context awareness"""
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
            
            # Business-context aware queries
            customer_col = business_context["entity_mappings"].get("customers")
            revenue_col = business_context["entity_mappings"].get("sales")
            
            if customer_col and customer_col in [c['name'] for c in info['columns']]:
                queries.extend([
                    f"SELECT DISTINCT {customer_col} FROM {table_name} {limit_clause};",
                    f"SELECT {customer_col}, COUNT(*) FROM {table_name} GROUP BY {customer_col} ORDER BY COUNT(*) DESC {limit_clause};"
                ])
            
            if revenue_col and revenue_col in [c['name'] for c in info['columns']]:
                queries.extend([
                    f"SELECT SUM({revenue_col}) as total_revenue FROM {table_name};",
                    f"SELECT AVG({revenue_col}) as avg_revenue FROM {table_name};"
                ])
                
                if customer_col and customer_col in [c['name'] for c in info['columns']]:
                    queries.extend([
                        f"SELECT {customer_col}, SUM({revenue_col}) as total FROM {table_name} GROUP BY {customer_col} ORDER BY total DESC {limit_clause};",
                        f"SELECT {customer_col}, AVG({revenue_col}) as avg_revenue FROM {table_name} GROUP BY {customer_col} ORDER BY avg_revenue DESC {limit_clause};"
                    ])
            
            # Add other business-aware patterns...
            for col in info['numeric_cols'][:3]:  # Top 3 numeric columns
                if col in business_context["metric_columns"]:
                    queries.extend([
                        f"SELECT {col}, COUNT(*) FROM {table_name} GROUP BY {col} ORDER BY COUNT(*) DESC {limit_clause};",
                        f"SELECT AVG({col}) as avg_{col}, MIN({col}) as min_{col}, MAX({col}) as max_{col} FROM {table_name};"
                    ])
            
            for query in queries:
                examples.append({
                    'type': 'sql',
                    'content': query
                })
        
        return examples

    def _generate_aggregation_patterns(self, tables_info: Dict, db_type: str, business_context: Dict) -> List[Dict]:
        """Generate aggregation patterns with business context"""
        examples = []
        limit_clause = "TOP 15" if db_type == 'mssql' else "LIMIT 15"
        
        for table, info in tables_info.items():
            table_name = info['full_name']
            
            # Get business-relevant columns
            customer_col = business_context["entity_mappings"].get("customers")
            revenue_col = business_context["entity_mappings"].get("sales")
            
            # Generate business-aware aggregation patterns
            if customer_col and revenue_col:
                if (customer_col in [c['name'] for c in info['columns']] and 
                    revenue_col in [c['name'] for c in info['columns']]):
                    
                    business_queries = [
                        f"SELECT {customer_col}, SUM({revenue_col}) as total_revenue, COUNT(*) as transaction_count FROM {table_name} GROUP BY {customer_col} ORDER BY total_revenue DESC {limit_clause};",
                        f"SELECT {customer_col}, AVG({revenue_col}) as avg_revenue_per_transaction FROM {table_name} GROUP BY {customer_col} HAVING COUNT(*) > 1 ORDER BY avg_revenue_per_transaction DESC {limit_clause};",
                        f"SELECT {customer_col}, SUM({revenue_col}) as total, MIN({revenue_col}) as min_transaction, MAX({revenue_col}) as max_transaction FROM {table_name} GROUP BY {customer_col} ORDER BY total DESC {limit_clause};"
                    ]
                    
                    for query in business_queries:
                        examples.append({
                            'type': 'sql',
                            'content': query
                        })
            
            # Continue with other aggregation patterns...
            for numeric_col in info['numeric_cols'][:2]:
                if numeric_col in business_context["metric_columns"]:
                    for group_col in info['text_cols'][:2]:
                        if group_col in business_context["dimension_columns"]:
                            examples.append({
                                'type': 'sql',
                                'content': f"SELECT {group_col}, SUM({numeric_col}) as total_{numeric_col}, COUNT(*) as count FROM {table_name} GROUP BY {group_col} ORDER BY total_{numeric_col} DESC {limit_clause};"
                            })
        
        return examples

    def _generate_dynamic_business_patterns(self, tables_info: Dict, db_type: str, business_context: Dict) -> List[Dict]:
        """Generate business logic patterns based on actual business context"""
        examples = []
        
        # Generate documentation based on discovered patterns
        business_docs = [
            f"This database contains {len(tables_info)} main tables for business analysis",
            f"Primary business entities identified: {', '.join(business_context['entity_mappings'].keys())}",
            f"Key metrics available for analysis: {', '.join(business_context['metric_columns'][:10])}",
            f"Main dimensions for grouping: {', '.join(business_context['dimension_columns'][:10])}"
        ]
        
        # Add table-specific business context
        for table, purpose in business_context["table_purposes"].items():
            business_docs.append(f"Table {table} is used for {purpose.replace('_', ' ')} analysis")
        
        # Add entity mapping rules
        for logical, physical in business_context["entity_mappings"].items():
            business_docs.append(f"When analyzing {logical}, use the {physical} column")
        
        # Add common business analysis patterns
        business_docs.extend([
            "Revenue analysis typically requires grouping by customer/client and summing revenue columns",
            "Performance analysis often involves calculating ratios and percentages between metrics",
            "Time-based analysis uses date columns for trending and period comparisons",
            "Top N analysis uses ORDER BY DESC with LIMIT clauses for rankings",
            "Comparative analysis uses aggregation functions like SUM, AVG, COUNT",
            "Business users often need both summary totals and detailed breakdowns"
        ])
        
        for doc in business_docs:
            examples.append({
                'type': 'documentation',
                'content': doc
            })
        
        return examples

    # Include all the other methods from the previous version...
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

    def _train_in_batches(self, examples: List[Dict], batch_size: int) -> Dict:
        """Train with all examples in batches with enhanced validation"""
        stats = {
            'total_examples': len(examples),
            'ddl_count': len([e for e in examples if e['type'] == 'ddl']),
            'sql_count': len([e for e in examples if e['type'] == 'sql']),
            'doc_count': len([e for e in examples if e['type'] == 'documentation']),
            'question_sql_count': len([e for e in examples if e['type'] == 'question_sql']),
            'batches_processed': 0,
            'successful_examples': 0,
            'errors': []
        }
        
        print(f"🔥 Training with {stats['total_examples']} examples:")
        print(f"   📋 DDL: {stats['ddl_count']}")
        print(f"   🔍 SQL: {stats['sql_count']}")  
        print(f"   📚 Documentation: {stats['doc_count']}")
        print(f"   ❓ Question-SQL pairs: {stats['question_sql_count']}")
        
        # Train in batches
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i+batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(examples) + batch_size - 1) // batch_size
            
            print(f"🔥 Processing batch {batch_num}/{total_batches} ({len(batch)} examples)")
            
            try:
                for example in batch:
                    try:
                        if example['type'] == 'ddl':
                            self.train(ddl=example['content'])
                        elif example['type'] == 'sql':
                            self.train(sql=example['content'])
                        elif example['type'] == 'documentation':
                            self.train(documentation=example['content'])
                        elif example['type'] == 'question_sql':
                            self.train(question=example['question'], sql=example['sql'])
                        
                        stats['successful_examples'] += 1
                    except Exception as e:
                        error_msg = f"Example error: {str(e)}"
                        stats['errors'].append(error_msg)
                        print(f"⚠️  {error_msg}")
                
                stats['batches_processed'] += 1
                print("✅", end="", flush=True)
                
            except Exception as e:
                error_msg = f"Batch {batch_num} error: {str(e)}"
                stats['errors'].append(error_msg)
                print(f"❌ {error_msg}")
        
        print(f"\n🎉 Completed {stats['batches_processed']}/{total_batches} batches")
        print(f"✅ Successfully trained: {stats['successful_examples']} examples")
        
        # Verify training data was stored
        self._verify_training_data()
        
        return stats

    def _verify_training_data(self):
        """Verify that training data was actually stored"""
        try:
            training_data = self.get_training_data()
            if training_data is not None:
                count = len(training_data)
                print(f"✅ Verified: {count} training examples stored in ChromaDB")
                
                # Check for specific business context
                if hasattr(training_data, 'to_dict'):
                    data_dict = training_data.to_dict('records')
                else:
                    data_dict = training_data if isinstance(training_data, list) else []
                
                # Look for our business context examples
                customer_examples = [item for item in data_dict if