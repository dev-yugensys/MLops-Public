"""
Data Preparation Page

This page allows users to upload and preprocess datasets for model training.
"""

import os
import streamlit as st
from pathlib import Path
import pandas as pd
import time
from preprocessing import (
    list_available_datasets,
    get_dataset_info,
    preprocess_iris_data,
    preprocess_housing_data,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR
)

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Page config
st.set_page_config(
    page_title="Data Preparation | MLOps Platform",
    page_icon="📊",
    layout="wide"
)

def show_dataset_info(file_path: str):
    """Display detailed information about a dataset"""
    with st.spinner('Analyzing dataset...'):
        info = get_dataset_info(file_path)
        if not info:
            st.error("❌ Could not read dataset information.")
            return
        
        st.success("✅ Dataset loaded successfully!")
        
        # Basic info in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Rows", f"{info['rows']:,}")
        with col2:
            st.metric("📋 Columns", info['columns'])
        with col3:
            missing_total = sum(info['missing_values'].values())
            st.metric("⚠️ Missing Values", f"{missing_total:,}")
        
        # Data types and sample in tabs
        tab1, tab2, tab3 = st.tabs(["📝 Data Types", "🔍 Sample Data", "📈 Missing Values"])
        
        with tab1:
            st.write("**Data Types**")
            dtype_df = pd.DataFrame({
                'Column': list(info['data_types'].keys()),
                'Data Type': [str(t) for t in info['data_types'].values()]
            })
            st.dataframe(
                dtype_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Column': st.column_config.TextColumn(width='medium'),
                    'Data Type': st.column_config.TextColumn(width='small')
                }
            )
        
        with tab2:
            st.write(f"**First 5 rows**")
            st.dataframe(
                info['sample'],
                use_container_width=True,
                hide_index=True
            )
        
        with tab3:
            if missing_total > 0:
                missing_series = pd.Series(info['missing_values'])
                missing_series = missing_series[missing_series > 0].sort_values(ascending=False)
                
                col1, col2 = st.columns([2, 3])
                with col1:
                    st.write("**Missing Values by Column**")
                    st.dataframe(
                        missing_series.rename('Missing Count'),
                        use_container_width=True
                    )
                with col2:
                    st.write("**Missing Values Distribution**")
                    st.bar_chart(missing_series)
            else:
                st.success("✅ No missing values found in the dataset!")

def process_dataset(file_name: str):
    """Process a dataset and save the preprocessed version"""
    input_path = RAW_DATA_DIR / file_name
    output_path = PROCESSED_DATA_DIR / f"preprocessed_{file_name}"
    
    with st.spinner(f'Processing {file_name}...'):
        try:
            # Show processing animation
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate processing steps
            status_text.text("Reading file...")
            progress_bar.progress(10)
            time.sleep(0.5)
            
            # Determine preprocessing function based on file name
            if "iris" in file_name.lower():
                status_text.text("Preprocessing iris data...")
                df, _ = preprocess_iris_data(str(input_path), str(output_path))
            elif "housing" in file_name.lower():
                status_text.text("Preprocessing housing data...")
                df = preprocess_housing_data(str(input_path), str(output_path))
            else:
                status_text.text("Using default preprocessing...")
                # Default preprocessing for unknown datasets
                df = pd.read_csv(input_path) if str(input_path).endswith('.csv') else pd.read_excel(input_path)
                df = df.dropna()
                df.to_csv(output_path, index=False)
            
            progress_bar.progress(80)
            time.sleep(0.5)
            
            # Show success message
            status_text.text("Processing complete!")
            progress_bar.progress(100)
            time.sleep(0.3)
            
            st.success(f"✅ Successfully processed and saved to: {output_path}")
            
            # Show preview of processed data
            st.subheader("Processed Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Add download button
            with open(output_path, 'rb') as f:
                st.download_button(
                    label="⬇️ Download Processed Data",
                    data=f,
                    file_name=output_path.name,
                    mime='text/csv' if str(output_path).endswith('.csv') else 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)  # Show detailed error in debug mode
            return False
    
    try:
        if "iris" in file_name.lower():
            df, saved_path = preprocess_iris_data(str(input_path), str(output_path))
        elif "housing" in file_name.lower():
            df, saved_path = preprocess_housing_data(str(input_path), str(output_path))
        else:
            st.error("Unsupported dataset type. Currently supports 'iris' and 'housing' datasets.")
            return
            
        if saved_path:
            st.success(f"✅ Dataset processed and saved to: {saved_path}")
            st.session_state.processed_file = saved_path
            
            # Show sample of processed data
            st.subheader("Processed Data Sample")
            st.dataframe(df.head())
            
            # Add download button
            with open(saved_path, "rb") as f:
                st.download_button(
                    label="Download Processed Data",
                    data=f,
                    file_name=os.path.basename(saved_path),
                    mime="text/csv" if saved_path.endswith('.csv') else "application/vnd.ms-excel"
                )
    except Exception as e:
        st.error(f"Error processing dataset: {str(e)}")

def main():
    st.title("📊 Data Preparation")
    st.caption("Upload, explore, and preprocess your datasets for machine learning")
    
    # File upload section
    st.header("Upload New Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"]
    )
    
    if uploaded_file is not None:
        try:
            file_path = RAW_DATA_DIR / uploaded_file.name
            with st.spinner(f"Saving {uploaded_file.name}..."):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success(f"✅ Saved {uploaded_file.name} to {file_path}")
            st.rerun()  # Refresh the page to show the new dataset
        except Exception as e:
            st.error(f"❌ Error saving file: {str(e)}")
    
    # Dataset selection and processing
    st.header("🔍 Dataset Explorer")
    datasets = list_available_datasets()
    
    if not datasets:
        st.info("ℹ️ No datasets found. Please upload a dataset to get started.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_dataset = st.selectbox(
            "Select a dataset to explore",
            datasets,
            index=0,
            format_func=lambda x: f"📁 {x}",
            help="Select a dataset to view its details and preprocess it"
        )
    
    with col2:
        st.write("")
        process_btn = st.button(
            "⚙️ Process Dataset",
            type="primary",
            disabled=not selected_dataset,
            help="Preprocess the selected dataset"
        )
    
    if selected_dataset:
        file_path = RAW_DATA_DIR / selected_dataset
        
        # Show dataset info in tabs
        tab1, tab2 = st.tabs(["📊 Dataset Info", "⚡ Quick Actions"])
        
        with tab1:
            show_dataset_info(str(file_path))
        
        with tab2:
            st.write("### Quick Actions")
            
            if process_btn or st.session_state.get('process_clicked', False):
                st.session_state['process_clicked'] = True
                process_dataset(selected_dataset)
            
            # Add more quick actions here
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Copy Path", help="Copy file path to clipboard"):
                    st.session_state.copied = True
                    st.rerun()
                
                if st.button("🗑️ Delete", help="Remove this dataset"):
                    try:
                        file_path.unlink()
                        st.success(f"✅ Deleted {selected_dataset}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error deleting file: {str(e)}")
            
            with col2:
                with open(file_path, 'rb') as f:
                    st.download_button(
                        "💾 Download",
                        data=f,
                        file_name=selected_dataset,
                        mime='text/csv' if selected_dataset.endswith('.csv') else 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        help="Download this dataset"
                    )
    
    # Processed files section
    st.header("📂 Processed Datasets")
    processed_files = list(PROCESSED_DATA_DIR.glob("*"))
    
    if not processed_files:
        st.info("ℹ️ No processed datasets found. Process a dataset to see it here.")
    else:
        st.write("Your processed datasets:")
        
        for file in sorted(processed_files, key=os.path.getmtime, reverse=True):
            with st.expander(f"📄 {file.name}"):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.caption(f"Last modified: {time.ctime(os.path.getmtime(file))}")
                    st.caption(f"Size: {file.stat().st_size / 1024:.1f} KB")
                
                with col2:
                    with open(file, 'rb') as f:
                        st.download_button(
                            "Download",
                            data=f,
                            file_name=file.name,
                            mime='text/csv' if str(file).endswith('.csv') else 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            key=f"dl_{file.name}",
                            use_container_width=True
                        )
                
                with col3:
                    if st.button("Delete", key=f"del_{file.name}", use_container_width=True):
                        try:
                            file.unlink()
                            st.success(f"✅ Deleted {file.name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error deleting file: {str(e)}")
                
                # Show a small preview of the processed file
                try:
                    df_preview = pd.read_csv(file) if str(file).endswith('.csv') else pd.read_excel(file)
                    st.dataframe(
                        df_preview.head(3),
                        use_container_width=True,
                        hide_index=True
                    )
                except Exception as e:
                    st.warning(f"Could not preview file: {str(e)}")

if __name__ == "__main__":
    main()
