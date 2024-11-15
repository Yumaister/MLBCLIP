import mlb_dataset as mlb

# Example of downloading videos
download_results = mlb.download_all_videos("/mnt/e/MML_RESEARCH/manifest/mlb-youtube-segmented.json", "/mnt/e/MML_RESEARCH/DATA")
mlb.extract_segmented_clips("/mnt/e/MML_RESEARCH/manifest/mlb-youtube-segmented.json", "/mnt/e/MML_RESEARCH/DATA", "/mnt/e/MML_RESEARCH/SEGMENTED")
mlb.extract_continuous_clips("/mnt/e/MML_RESEARCH/manifest/mlb-youtube-segmented.json", "/mnt/e/MML_RESEARCH/DATA", "/mnt/e/MML_RESEARCH/CONTINUOUS")