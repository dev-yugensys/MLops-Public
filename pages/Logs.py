import streamlit as st
import pandas as pd
from datetime import datetime
import json
from utils.db import get_requests, get_request_stats, delete_request

st.set_page_config(page_title="Logs", layout="wide")

def format_json(data: str) -> str:
    """Format JSON string for better readability."""
    if not data:
        return ""
    try:
        parsed = json.loads(data)
        return json.dumps(parsed, indent=2)
    except:
        return data

def main():
    st.title("Model Request Logs")
    
    # Get and display stats
    stats = get_request_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Requests", stats['total'])
    with col2:
        st.metric("Successful", stats['by_status'].get('completed', 0))
    with col3:
        st.metric("Failed", stats['by_status'].get('failed', 0))
    
    # Filter options
    st.sidebar.header("Filters")
    user_id = st.sidebar.text_input("Filter by User ID")
    limit = st.sidebar.slider("Max entries", 10, 1000, 100)
    
    # Get and display requests
    requests = get_requests(user_id=user_id if user_id else None, limit=limit)
    
    if not requests:
        st.info("No requests found matching the criteria.")
        return
    
    st.subheader("Recent Requests")
    
    # Create a DataFrame with all the data
    df = pd.DataFrame([{
        'ID': r['id'],
        'Timestamp': r['timestamp'],
        'User': r['user_id'],
        'Model': r['model_name'],
        'Status': r['status'].capitalize(),
        'Input Size': len(r['input_data']) if r['input_data'] else 0,
        'Has Output': 'Yes' if r['output_data'] else 'No',
        'Delete': f"Delete {r['id']}"  # This will be used for the button label
    } for r in requests])
    
    # Display the table with a delete button in each row
    for i, row in df.iterrows():
        cols = st.columns([1, 2, 2, 2, 1, 1, 1, 1])  # Adjust column widths as needed
        
        # Display row data
        with cols[0]:
            st.text(row['ID'])
        with cols[1]:
            st.text(str(row['Timestamp']))
        with cols[2]:
            st.text(row['User'])
        with cols[3]:
            st.text(row['Model'])
        with cols[4]:
            st.text(row['Status'])
        with cols[5]:
            st.text(row['Input Size'])
        with cols[6]:
            st.text(row['Has Output'])
        
        # Add delete button in the last column
        with cols[7]:
            if st.button('🗑️', key=f"del_{row['ID']}"):
                if delete_request(row['ID']):
                    st.success(f"Deleted request #{row['ID']}")
                    st.rerun()
                else:
                    st.error(f"Failed to delete request #{row['ID']}")
    
    # Add some spacing
    st.write("")
    
    # Store the requests for the details view
    df = pd.DataFrame([{
        'ID': r['id'],
        'Timestamp': r['timestamp'],
        'User': r['user_id'],
        'Model': r['model_name'],
        'Status': r['status'].capitalize(),
        'Input Size': len(r['input_data']) if r['input_data'] else 0,
        'Has Output': 'Yes' if r['output_data'] else 'No'
    } for r in requests])
    
    # Show details for selected request
    if not df.empty:
        selected_id = st.selectbox("Select a request to view details", [""] + df['ID'].astype(str).tolist())
        
        if selected_id:
            selected = next((r for r in requests if r['id'] == int(selected_id)), None)
            if selected:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Input Data")
                    st.code(format_json(selected['input_data']), language='json')
                
                with col2:
                    st.subheader("Output Data")
                    if selected['output_data']:
                        st.code(format_json(selected['output_data']), language='json')
                    else:
                        st.warning("No output data available")
                    
                    if selected['error_message']:
                        st.error(f"Error: {selected['error_message']}")

if __name__ == "__main__":
    main()
