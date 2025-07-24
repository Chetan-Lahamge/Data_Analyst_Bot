import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import BytesIO
import base64
import json
from Agents.data_catalogue_agent import get_db_connection
from Agents.data_extractor_agent import DataExtractorAgent
import config
from openai import OpenAI

class DatabaseEDA:
    def __init__(self, output_dir="EDA"):
        """Initialize the DatabaseEDA class with database connection"""
        self.connection = get_db_connection()
        self.data_extractor = DataExtractorAgent(self.connection)
        self.output_dir = output_dir
        
        # Create directory structure
        self.images_dir = os.path.join(output_dir, "images")
        self.summaries_dir = os.path.join(output_dir, "summaries")
        self.stats_dir = os.path.join(output_dir, "statistics")
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        
    def get_all_tables(self):
        """Get all table names from the database"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'")
        return cursor.fetchall()
    
    def table_to_dataframe(self, schema_name, table_name, limit=10000):
        """Convert a table to a pandas DataFrame with a reasonable row limit"""
        query = f"SELECT TOP {limit} * FROM [{schema_name}].[{table_name}]"
        try:
            df = pd.read_sql(query, self.connection)
            return df
        except Exception as e:
            print(f"Error loading table {schema_name}.{table_name}: {e}")
            return None
    
    def analyze_numeric_columns(self, df, schema_name, table_name):
        """Generate analysis for numeric columns"""
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) == 0:
            return "No numeric columns found.", []
        
        table_image_dir = os.path.join(self.images_dir, f"{schema_name}_{table_name}")
        os.makedirs(table_image_dir, exist_ok=True)
        
        image_paths = []
        
        # Univariate analysis
        for col in numeric_cols:
            try:
                if df[col].isna().all() or len(df[col].dropna()) == 0:
                    print(f"Skipping {col} - all values are NaN")
                    continue
                    
                plt.figure(figsize=(10, 6))
                plt.subplot(1, 2, 1)
                sns.histplot(df[col].dropna(), kde=True)
                plt.title(f'Distribution of {col}')
                
                plt.subplot(1, 2, 2)
                sns.boxplot(y=df[col].dropna())
                plt.title(f'Boxplot of {col}')
                
                plt.tight_layout()
                img_path = os.path.join(table_image_dir, f"numeric_{col}.png")
                plt.savefig(img_path)
                plt.close()
                image_paths.append(img_path)
            except Exception as e:
                print(f"Error analyzing numeric column {col}: {e}")
                plt.close()  # Ensure figure is closed
        
        # Correlation matrix if multiple numeric columns
        if len(numeric_cols) > 1:
            try:
                # Check if we have at least some non-NaN data
                corr_data = df[numeric_cols].dropna(how='all')
                if not corr_data.empty and not corr_data.isna().all().all():
                    plt.figure(figsize=(12, 10))
                    corr_matrix = corr_data.corr()
                    # Replace NaN values in correlation matrix with 0
                    corr_matrix = corr_matrix.fillna(0)
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
                    plt.title(f'Correlation Matrix: {schema_name}.{table_name}')
                    plt.tight_layout()
                    img_path = os.path.join(table_image_dir, "correlation_matrix.png")
                    plt.savefig(img_path)
                    plt.close()
                    image_paths.append(img_path)
            except Exception as e:
                print(f"Error creating correlation matrix: {e}")
                plt.close()  # Ensure figure is closed
        
        return table_image_dir, image_paths
    
    def analyze_categorical_columns(self, df, schema_name, table_name):
        """Analyze categorical columns in the dataframe"""
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        if not cat_cols:
            print(f"No categorical columns found in {schema_name}.{table_name}")
            return None, []
        
        # Create a directory for this table
        table_images_dir = os.path.join(self.images_dir, f"{schema_name}_{table_name}")
        os.makedirs(table_images_dir, exist_ok=True)
        
        # Stats file path
        stats_file = os.path.join(self.stats_dir, f"{schema_name}_{table_name}_categorical_stats.json")
        
        cat_stats = {}
        image_paths = []
        
        for i, column in enumerate(cat_cols):
            try:
                # Create a subplot
                plt.figure(figsize=(10, 6))
                
                # Truncate long category values to prevent rendering errors
                value_counts = df[column].value_counts().head(10)
                labels = [str(x)[:20] + '...' if len(str(x)) > 20 else str(x) for x in value_counts.index]
                
                # Simple bar plot with truncated labels
                plt.bar(range(len(labels)), value_counts.values)
                plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
                plt.title(f'Top 10 values for {column}')
                plt.tight_layout()
                
                # Save the figure
                img_path = os.path.join(table_images_dir, f"{column}_distribution.png")
                plt.savefig(img_path)
                plt.close()
                
                image_paths.append(img_path)
            except Exception as e:
                print(f"Skipping plot for {column}: {str(e)}")
                continue
        
        # Save the statistics to a JSON file
        with open(stats_file, 'w') as f:
            json.dump(cat_stats, f, indent=4)
        
        return stats_file, image_paths
    
    def analyze_datetime_columns(self, df, schema_name, table_name):
        """Generate analysis for datetime columns"""
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) == 0:
            return "No datetime columns found.", []
        
        table_image_dir = os.path.join(self.images_dir, f"{schema_name}_{table_name}")
        os.makedirs(table_image_dir, exist_ok=True)
        
        image_paths = []
        
        for col in datetime_cols:
            try:
                # Skip if all values are NaN
                if df[col].isna().all() or len(df[col].dropna()) == 0:
                    print(f"Skipping datetime column {col} - all values are NaN")
                    continue
                
                # Time series plot
                plt.figure(figsize=(15, 5))
                df[col].dropna().value_counts().sort_index().plot()
                plt.title(f'Time distribution of {col}')
                plt.tight_layout()
                img_path = os.path.join(table_image_dir, f"time_{col}.png")
                plt.savefig(img_path)
                plt.close()
                image_paths.append(img_path)
                
                # Extract time components for further analysis
                if len(df) > 0 and len(df[col].dropna()) > 0:
                    plt.figure(figsize=(12, 8))
                    month_data = df[col].dropna().dt.month.value_counts().sort_index()
                    if not month_data.empty:
                        month_data.plot(kind='bar')
                        plt.title(f'Month distribution of {col}')
                        plt.tight_layout()
                        img_path = os.path.join(table_image_dir, f"month_{col}.png")
                        plt.savefig(img_path)
                        plt.close()
                        image_paths.append(img_path)
                    else:
                        plt.close()
            except Exception as e:
                print(f"Error analyzing datetime column {col}: {e}")
                plt.close()  # Ensure figure is closed
        
        return table_image_dir, image_paths
    
    def encode_image(self, image_path):
        """Encode image to base64 for OpenAI API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def generate_vision_summary(self, image_paths, schema_name, table_name, df):
        """Generate EDA-focused summary using GPT-4o Vision focusing on patterns and insights"""
        if not image_paths:
            return "No visualizations available for analysis."
            
        # Prepare column info
        column_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample = str(df[col].iloc[0]) if len(df) > 0 else "N/A"
            null_pct = df[col].isna().mean() * 100
            unique_count = df[col].nunique()
            
            # Additional statistics for numeric columns
            stats = {}
            if df[col].dtype in ['int64', 'float64']:
                if not df[col].isna().all():
                    stats = {
                        "min": float(df[col].min()) if not pd.isna(df[col].min()) else "N/A",
                        "max": float(df[col].max()) if not pd.isna(df[col].max()) else "N/A",
                        "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else "N/A",
                        "median": float(df[col].median()) if not pd.isna(df[col].median()) else "N/A",
                        "std_dev": float(df[col].std()) if not pd.isna(df[col].std()) else "N/A"
                    }
            
            column_info.append({
                "name": col,
                "type": dtype,
                "sample": sample,
                "null_percent": f"{null_pct:.2f}%",
                "unique_values": unique_count,
                "stats": stats
            })
            
        # Use the exact prompt from the user's example
        custom_prompt = '''
        You are a DATA ANALYST tasked with analyzing data visualizations.
        Generate a detailed summary in 8-10 lines, highlighting the following:
        1. Title and axes information, 
        2. Categories and their respective counts, 
        3. Any clear patterns or trends,
        4. Comparison between categories, 
        5. Any additional insights that can be inferred from the differences.
        Make sure the summary is detailed and covers all aspects clearly.
        Avoid using any special characters, symbols, or markdown formatting in the output.
        DO NOT USE ANY symbols, or markdown formatting. The summary should be clear, detailed,
        and focused on key details like categories, counts, and patterns. 
        '''
        
        # Create a list of messages, starting with up to 5 images (to avoid token limits)
        image_paths_to_use = image_paths[:5]  # Limit to 5 images to avoid token limits
        
        messages = [{"role": "system", "content": custom_prompt}]
        
        # Add table context for better understanding
        context_text = f"""
        Analyzing visualizations for database table: {schema_name}.{table_name}
        
        Table Information:
        - Rows: {len(df)}
        - Columns: {len(df.columns)}
        
        Focus on examining distributions, patterns, and trends in the data.
        """
        
        # Add the user message with context and images
        user_content = [
            {
                "type": "text", 
                "text": context_text
            }
        ]
        
        # Add images to the content
        for img_path in image_paths_to_use:
            base64_image = self.encode_image(img_path)
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "high"
                }
            })
        
        messages.append({"role": "user", "content": user_content})
        
        try:
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=2500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating vision summary: {e}")
            return f"Error generating summary: {str(e)}"
    
    def generate_dataframe_skeleton(self, df, schema_name, table_name):
        """Generate a detailed skeleton of the DataFrame with statistics"""
        skeleton_path = os.path.join(self.stats_dir, f"{schema_name}_{table_name}_skeleton.txt")
        
        with open(skeleton_path, 'w') as f:
            f.write(f"Detailed Table Skeleton for {schema_name}.{table_name}\n")
            f.write("=" * 50 + "\n\n")
            
            for column in df.columns:
                f.write(f"Column: {column}\n")
                f.write(f"Data Type: {df[column].dtype}\n")
                f.write(f"Number of Unique Values: {df[column].nunique()}\n")
                f.write(f"Missing Values: {df[column].isna().sum()} ({df[column].isna().mean()*100:.2f}%)\n")
                
                if pd.api.types.is_numeric_dtype(df[column]):
                    f.write("Statistics:\n")
                    f.write(df[column].describe().to_string() + "\n\n")
                else:
                    f.write("Top Categories:\n")
                    f.write(df[column].value_counts().head(10).to_string() + "\n\n")
                    
                f.write("-" * 50 + "\n\n")
        
        return skeleton_path
    
    def analyze_correlations(self, df, schema_name, table_name):
        """Analyze correlations between numeric columns"""
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_cols) <= 1:
            return None, []
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr().fillna(0)
        
        # Save correlation matrix as CSV
        corr_path = os.path.join(self.stats_dir, f"{schema_name}_{table_name}_correlations.csv")
        corr_matrix.to_csv(corr_path)
        
        # Create heatmap visualization
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title(f'Correlation Matrix: {schema_name}.{table_name}')
        plt.tight_layout()
        
        # Save correlation heatmap
        img_path = os.path.join(self.images_dir, f"{schema_name}_{table_name}", "correlation_heatmap.png")
        plt.savefig(img_path)
        plt.close()
        
        return corr_path, img_path
    
    def run_eda_for_table(self, schema_name, table_name):
        """Run complete EDA for a single table"""
        print(f"Analyzing {schema_name}.{table_name}...")
        
        # Load data
        df = self.table_to_dataframe(schema_name, table_name)
        if df is None or df.empty:
            print(f"Skipping {schema_name}.{table_name} - empty or error loading table")
            return None
        
        # Generate EDA outputs with image paths
        num_path, num_images = self.analyze_numeric_columns(df, schema_name, table_name)
        cat_path, cat_images = self.analyze_categorical_columns(df, schema_name, table_name)
        time_path, time_images = self.analyze_datetime_columns(df, schema_name, table_name)
        
        # Generate correlations (new)
        corr_path, corr_img = self.analyze_correlations(df, schema_name, table_name) if len(df.select_dtypes(include=['int64', 'float64']).columns) > 1 else (None, None)
        
        # Generate skeleton (new)
        skeleton_path = self.generate_dataframe_skeleton(df, schema_name, table_name)
        
        # Collect all image paths
        all_images = num_images + cat_images + time_images
        if corr_img:
            all_images.append(corr_img)
        
        # Generate summary statistics
        stats_path = os.path.join(self.stats_dir, f"{schema_name}_{table_name}_stats.csv")
        df.describe(include='all').to_csv(stats_path)
        
        # Generate LLM summary using GPT-4o Vision
        print(f"Generating GPT-4o Vision summary for {schema_name}.{table_name}...")
        vision_summary = self.generate_vision_summary(all_images, schema_name, table_name, df)
        
        # Save the summary as txt file
        summary_path = os.path.join(self.summaries_dir, f"{schema_name}_{table_name}_eda_insights.txt")
        with open(summary_path, 'w') as f:
            f.write(f"EDA INSIGHTS FOR {schema_name}.{table_name}\n")
            f.write("="*50 + "\n\n")
            f.write(vision_summary)
        
        return {
            "table_name": f"{schema_name}.{table_name}",
            "row_count": len(df),
            "column_count": len(df.columns),
            "images_dir": os.path.join(self.images_dir, f"{schema_name}_{table_name}"),
            "stats_path": stats_path,
            "summary_path": summary_path,
            "skeleton_path": skeleton_path,
            "correlation_path": corr_path
        }
    
    def run_eda_for_all_tables(self):
        """Run EDA for all tables in the database"""
        tables = self.get_all_tables()
        results = []
        
        for schema, table in tables:
            result = self.run_eda_for_table(schema, table)
            if result:
                results.append(result)
        
        return results
    
    def generate_eda_report(self, results):
        """Generate a summary report of all EDA results"""
        report_path = os.path.join(self.output_dir, "eda_insights_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("DATABASE EDA INSIGHTS REPORT\n")
            f.write("============================\n\n")
            
            for result in results:
                f.write(f"TABLE: {result['table_name']}\n")
                f.write(f"Rows: {result['row_count']} | Columns: {result['column_count']}\n")
                f.write(f"EDA Insights: {result['summary_path']}\n")
                f.write(f"Statistics: {result['stats_path']}\n")
                f.write(f"Visualizations: {result['images_dir']}\n\n")
                
                # Include a snippet of the summary
                try:
                    with open(result['summary_path'], 'r') as summary_file:
                        summary_content = summary_file.read()
                        preview_lines = summary_content.split('\n')[:5]
                        preview = '\n'.join(preview_lines) + "...\n"
                        f.write("PREVIEW:\n")
                        f.write(f"{preview}\n")
                except Exception as e:
                    f.write(f"Error reading summary: {e}\n")
                
                f.write("----------------------------\n\n")
        
        return report_path


if __name__ == "__main__":
    # Run the EDA process
    eda = DatabaseEDA()
    results = eda.run_eda_for_all_tables()
    report_path = eda.generate_eda_report(results)
    print(f"EDA completed. Report available at: {report_path}") 
