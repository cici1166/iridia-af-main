import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

def create_label_array(labels_df, total_length):
    """
    Create complete label array based on label file
    :param labels_df: Label DataFrame
    :param total_length: Total length of ECG data
    :return: Label array, 1 for AF, 0 for NSR
    """
    labels = np.zeros(total_length)
    for _, row in labels_df.iterrows():
        start_idx = row['start_qrs_index']
        end_idx = row['end_qrs_index']
        labels[start_idx:end_idx] = 1  # Mark AF segments as 1
    return labels

def extract_segments(ecg_data, labels, segment_length=120000):
    """
    Extract segments from ECG data
    :param ecg_data: ECG data, shape (length,)
    :param labels: Label data, shape (length,)
    :param segment_length: Segment length (10 minutes = 120000 points)
    :return: List of segments and corresponding label list
    """
    segments = []
    segment_labels = []
    
    # Check if there is any AF episode (global label is 1)
    has_af = np.any(labels == 1)
    
    if has_af:
        # Find all AF episodes
        # Get the indices where AF starts (transition from 0 to 1)
        af_starts = np.where(np.diff(labels.astype(int)) == 1)[0] + 1
        
        # For each AF episode
        for af_start in af_starts:
            # Check if there are enough points before AF (10 minutes)
            if af_start >= segment_length:
                # Check if the previous segment contains any AF
                prev_segment_labels = labels[af_start - segment_length:af_start]
                if not np.any(prev_segment_labels == 1):
                    # Extract pre-AF segment (10 minutes before AF episode)
                    start_idx = af_start - segment_length
                    end_idx = af_start
                    segment = ecg_data[start_idx:end_idx]
                    segments.append(segment)
                    segment_labels.append('pre_af')
    else:
        # Process NSR segments
        # Calculate the number of complete 10-minute segments that can be extracted
        num_segments = len(ecg_data) // segment_length
        
        # Extract all complete 10-minute segments
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length
            segment = ecg_data[start_idx:end_idx]
            segments.append(segment)
            segment_labels.append('nsr')
    
    return segments, segment_labels

def process_record(record_path, output_h5, output_csv, segment_length=120000):
    """
    Process a single record
    :param record_path: Path to the record
    :param output_h5: Output HDF5 file
    :param output_csv: Output CSV file
    :param segment_length: Segment length (10 minutes = 120000 points)
    """
    try:
        # Read ECG data
        ecg_file = os.path.join(record_path, os.path.basename(record_path) + '_ecg_00.h5')
        if not os.path.exists(ecg_file):
            print(f"Error: ECG file not found: {ecg_file}")
            return 0
            
        with h5py.File(ecg_file, 'r') as f:
            ecg_data = f['ecg'][:]
            # Use only the first lead
            ecg_data = ecg_data[:, 0]
        print(f"Reading ECG data: {ecg_file}")
        print(f"ECG data length: {len(ecg_data)}")
        print(f"ECG data shape: {ecg_data.shape}")
        
        # Read labels
        label_file = os.path.join(record_path, os.path.basename(record_path) + '_ecg_labels.csv')
        if not os.path.exists(label_file):
            print(f"Error: Label file not found: {label_file}")
            return 0
            
        try:
            labels_df = pd.read_csv(label_file)
            print(f"Label file columns: {labels_df.columns.tolist()}")
            # Create complete label array
            labels = create_label_array(labels_df, len(ecg_data))
        except Exception as e:
            print(f"Error: Failed to read label file: {str(e)}")
            return 0
            
        print(f"Record labels: {np.unique(labels)} (1=AF, 0=NSR)")
        print(f"Reading label file: {label_file}")
        print(f"Label length: {len(labels)}")
        
        # Ensure ECG data and labels have matching lengths
        if len(ecg_data) != len(labels):
            print(f"Error: ECG data length ({len(ecg_data)}) does not match label length ({len(labels)})")
            return 0
        
        # Extract segments
        segments, segment_labels = extract_segments(ecg_data, labels, segment_length)
        print(f"Extracted {len(segments)} segments")
        
        # Save segments
        record_id = os.path.basename(record_path)
        for i, (segment, label) in enumerate(zip(segments, segment_labels)):
            sample_id = f"{label}_{record_id}_{i:04d}"
            output_h5.create_dataset(sample_id, data=segment)
            output_csv.append({
                'sample_id': sample_id,
                'record_id': record_id,
                'patient_id': record_id.split('_')[1],
                'start_index': i * segment_length,
                'segment_label': label
            })
        
        return len(segments)
        
    except Exception as e:
        print(f"Error processing record: {str(e)}")
        import traceback
        print(f"Error details:\n{traceback.format_exc()}")
        return 0

def main():
    # Create output directory
    os.makedirs('dataset', exist_ok=True)
    
    # Create output files
    output_h5 = h5py.File('dataset/processed_data.h5', 'w')
    output_csv = []
    
    # Process all records
    records_dir = 'data/records'
    if not os.path.exists(records_dir):
        print(f"Error: Records directory not found: {records_dir}")
        return
        
    records = [d for d in os.listdir(records_dir) if os.path.isdir(os.path.join(records_dir, d))]
    total_records = len(records)
    sample_count = 0
    
    print(f"Starting to process {total_records} records...")
    
    for i, record in enumerate(tqdm(records), 1):
        record_path = os.path.join(records_dir, record)
        print(f"\nProcessing record {record} ({i}/{total_records})")
        try:
            samples = process_record(record_path, output_h5, output_csv)
            sample_count += samples
            print(f"Successfully processed {record}, extracted {samples} samples")
        except Exception as e:
            print(f"Error processing record {record}: {str(e)}")
    
    # Save metadata
    if output_csv:
        pd.DataFrame(output_csv).to_csv('dataset/metadata.csv', index=False)
        print(f"\nProcessing complete, extracted {sample_count} samples in total")
        print(f"Data saved to dataset/processed_data.h5 and dataset/metadata.csv")
    else:
        print("\nError: No samples were successfully extracted")

if __name__ == '__main__':
    main()


