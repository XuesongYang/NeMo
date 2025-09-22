#!/usr/bin/env python3
import glob
import json
import logging
import multiprocessing as mp
import os
import tarfile


def setup_process_logger(process_id, log_dir="./logs"):
    """Set up individual logger for each process"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Use PID to ensure unique logger names across processes
    logger_name = f"process_{process_id}_{os.getpid()}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create file handler for this process
    log_file = os.path.join(log_dir, f"process_{process_id:02d}_{os.getpid()}.log")
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    return logger


def load_target_ids(txt_file_path):
    """Load target IDs from the text file"""
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f)


def filter_tar_files_by_range(tar_files, start_num=0, end_num=22590):
    """Filter tar files to only include those within the specified numeric range"""
    # Sort tar files by filename (which will sort them numerically due to zero-padding)
    sorted_tar_files = sorted(tar_files)

    # Since files are numbered continuously from 000000, we can slice directly
    # start_num corresponds to index start_num, end_num corresponds to index end_num
    total_files = len(sorted_tar_files)
    actual_end = min(end_num, total_files - 1)  # Handle case where end_num exceeds available files

    if start_num >= total_files:
        print(f"Start number {start_num} exceeds total files {total_files}, no files to process")
        return []

    filtered_files = sorted_tar_files[start_num : actual_end + 1]

    print(f"Processing files from index {start_num} to {actual_end} (inclusive)")
    print(f"First file: {os.path.basename(filtered_files[0]) if filtered_files else 'None'}")
    print(f"Last file: {os.path.basename(filtered_files[-1]) if filtered_files else 'None'}")

    return filtered_files


def process_single_tar_file(tar_file_info):
    """Process a single tar file - designed for multiprocessing"""
    tar_file, target_ids, output_dir, process_id = tar_file_info
    
    # Set up logger for this process
    logger = setup_process_logger(process_id)
    
    logger.info(f"Starting processing: {os.path.basename(tar_file)}")
    
    extracted_json_count = 0
    extracted_flac_count = 0
    skipped_flac_count = 0
    
    try:
        with tarfile.open(tar_file, 'r') as tar:
            # Get all members
            members = tar.getmembers()
            
            # Separate JSON and FLAC files
            json_files = {m.name: m for m in members if m.name.endswith('.json')}
            flac_files = {m.name: m for m in members if m.name.endswith('.flac')}
            
            logger.info(f"Found {len(json_files)} JSON files and {len(flac_files)} FLAC files")
            
            # Always extract all JSON files
            for json_member in json_files.values():
                try:
                    tar.extract(json_member, path=output_dir)
                    extracted_json_count += 1
                    logger.debug(f"Extracted JSON: {json_member.name}")
                except (OSError, IOError) as e:
                    logger.warning(f"Failed to extract JSON {json_member.name}: {e}")
            
            # Check each FLAC file's corresponding JSON for ID match
            for flac_name, flac_member in flac_files.items():
                # Get basename without extension
                basename = os.path.splitext(flac_name)[0]
                json_name = basename + '.json'
                
                if json_name in json_files:
                    # Extract JSON temporarily to check ID
                    json_path = os.path.join(output_dir, json_name)
                    
                    try:
                        with open(json_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Check if ID matches target IDs
                        if 'id' in metadata and str(metadata['id']).removeprefix("context_") in target_ids:
                            try:
                                tar.extract(flac_member, path=output_dir)
                                extracted_flac_count += 1
                                logger.debug(f"Extracted FLAC: {flac_member.name} (ID: {metadata['id']})")
                            except (OSError, IOError) as e:
                                logger.warning(f"Failed to extract FLAC {flac_member.name}: {e}")
                                skipped_flac_count += 1
                        else:
                            skipped_flac_count += 1
                            logger.debug(f"Skipped FLAC: {flac_member.name} (ID not in target list)")
                    
                    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                        logger.error(f"Error reading JSON {json_name}: {e}")
                        skipped_flac_count += 1
                else:
                    logger.warning(f"No corresponding JSON found for FLAC: {flac_name}")
                    skipped_flac_count += 1
        
        logger.info(f"Completed {os.path.basename(tar_file)}: "
                   f"JSON={extracted_json_count}, FLAC extracted={extracted_flac_count}, "
                   f"FLAC skipped={skipped_flac_count}")
        
        return {
            'tar_file': os.path.basename(tar_file),
            'process_id': process_id,
            'extracted_json': extracted_json_count,
            'extracted_flac': extracted_flac_count,
            'skipped_flac': skipped_flac_count,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Error processing {os.path.basename(tar_file)}: {e}")
        return {
            'tar_file': os.path.basename(tar_file),
            'process_id': process_id,
            'error': str(e),
            'success': False
        }


def extract_selective_files(tar_dir, txt_file, output_dir, start_num=0, end_num=22590, num_processes=64):
    """Extract tar files selectively based on JSON metadata using multiprocessing"""
    target_ids = load_target_ids(txt_file)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logs directory
    os.makedirs("./logs", exist_ok=True)

    # Get all tar files and filter by range
    all_tar_files = glob.glob(os.path.join(tar_dir, "*.tar"))
    tar_files_to_process = filter_tar_files_by_range(all_tar_files, start_num=start_num, end_num=end_num)

    print(f"Found {len(all_tar_files)} total tar files, processing {len(tar_files_to_process)} files in range")
    print(f"Using {num_processes} parallel processes")

    if not tar_files_to_process:
        print("No tar files to process!")
        return

    # Prepare arguments for multiprocessing
    # Each process gets: (tar_file, target_ids, output_dir, process_id)
    process_args = []
    for i, tar_file in enumerate(tar_files_to_process):
        process_id = i % num_processes  # Distribute files across processes using round-robin
        process_args.append((tar_file, target_ids, output_dir, process_id))
    
    # Log the distribution
    files_per_process = {}
    for _, _, _, pid in process_args:
        files_per_process[pid] = files_per_process.get(pid, 0) + 1
    
    print(f"File distribution across processes:")
    for pid in sorted(files_per_process.keys()):
        print(f"  Process {pid:02d}: {files_per_process[pid]} files")

    print(f"Starting parallel processing of {len(tar_files_to_process)} tar files...")
    
    # Use multiprocessing to process tar files in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_single_tar_file, process_args)
    
    # Aggregate and report results
    successful_files = 0
    failed_files = 0
    total_json_extracted = 0
    total_flac_extracted = 0
    total_flac_skipped = 0
    
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    
    for result in results:
        if result['success']:
            successful_files += 1
            total_json_extracted += result['extracted_json']
            total_flac_extracted += result['extracted_flac']
            total_flac_skipped += result['skipped_flac']
            print(f"✓ {result['tar_file']} (Process {result['process_id']:02d}): "
                  f"JSON={result['extracted_json']}, FLAC={result['extracted_flac']}, "
                  f"Skipped={result['skipped_flac']}")
        else:
            failed_files += 1
            print(f"✗ {result['tar_file']} (Process {result['process_id']:02d}): "
                  f"ERROR - {result['error']}")
    
    print("="*80)
    print(f"TOTAL RESULTS:")
    print(f"  Successful files: {successful_files}")
    print(f"  Failed files: {failed_files}")
    print(f"  Total JSON files extracted: {total_json_extracted}")
    print(f"  Total FLAC files extracted: {total_flac_extracted}")
    print(f"  Total FLAC files skipped: {total_flac_skipped}")
    print(f"  Individual process logs saved in: ./logs/")
    print("="*80)


if __name__ == "__main__":
    # Ensure multiprocessing works correctly on all platforms
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method has already been set
        pass
    
    tar_directory = "./lhotse_shar_shuffle_shardSize256/context_audio"
    txt_file = "./context_audio_cut_ids.txt.exclusive"
    output_directory = "./context_audio"

    # Process tar files from recording.000000.tar to recording.022590.tar
    start_num = 0  # corresponds to recording.000000.tar
    end_num = 22590  # corresponds to recording.022590.tar
    num_processes = 64  # Use 64 cores for parallel processing (optimal for 96-core system)

    print(f"Starting parallel extraction with {num_processes} processes...")
    print(f"Processing tar files from index {start_num} to {end_num}")
    print(f"Target IDs file: {txt_file}")
    print(f"Output directory: {output_directory}")
    print("-" * 80)

    extract_selective_files(tar_directory, txt_file, output_directory, start_num, end_num, num_processes)
