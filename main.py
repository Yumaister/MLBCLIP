import os
import cv2
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torchvision import transforms
from typing import List, Tuple

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Specify the device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --- Step 1: Parse the JSON files to load video and caption data --- #

# Load caption data
with open("/mnt/e/MML_RESEARCH/manifest/mlb-youtube-captions.json") as f:
    caption_data = json.load(f)

# Define a helper function to load video annotations
def load_annotations(file_path: str) -> dict:
    with open(file_path) as f:
        return json.load(f)

# Load continuous and segmented annotations
continuous_data = load_annotations("/mnt/e/MML_RESEARCH/manifest/mlb-youtube-continuous.json")
segmented_data = load_annotations("/mnt/e/MML_RESEARCH/manifest/mlb-youtube-segmented.json")

# --- Step 2: Frame Extraction --- #
# Define a function to extract frames from a video between specific times
def extract_frames(video_path: str, output_dir: str, start_time: float, end_time: float, frame_rate: int = 1):
    """
    Extract frames from a video between `start_time` and `end_time` and save them in `output_dir`.
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps * frame_rate
    count = 0
    frame_list = []

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Move to the start frame
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000) > end_time:
            break
        # Save frame at the specified interval
        if count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_list.append(frame_path)
        count += 1
    
    cap.release()
    return frame_list

# --- Step 3: Prepare Dataset Class for Data Loading --- #

class VideoCaptionDataset(Dataset):
    def __init__(self, annotations: dict, caption_data: dict, video_dir: str, transform=None):
        """
        Dataset for loading video frames and captions.
        `annotations`: Dictionary with video annotation data.
        `caption_data`: Dictionary with caption data for each video.
        `video_dir`: Directory where videos are stored.
        `transform`: Optional transformations to apply to frames.
        """
        self.annotations = annotations
        self.caption_data = caption_data
        self.video_dir = video_dir
        self.transform = transform or transforms.ToTensor()
        self.processor = processor  # CLIP Processor to tokenize captions

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get video ID and caption for the current sample
        video_id = list(self.annotations.keys())[idx]
        video_info = self.annotations[video_id]
        
        # Get the start and end times and the captions
        start_time = video_info['start']
        end_time = video_info['end']
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        
        # Extract frames for this segment
        frame_dir = f"/tmp/{video_id}_frames"
        frames = extract_frames(video_path, frame_dir, start_time, end_time)
        
        # Load one frame for simplicity (or add a sampling strategy here if needed)
        image = Image.open(frames[5]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Get the corresponding caption
        caption = self.caption_data[video_id][0]['caption']  # Use the first caption in this example
        
        # Tokenize the caption and prepare inputs for CLIP
        inputs = self.processor(text=caption, images=image, return_tensors="pt", padding=True)
        return inputs

# Initialize Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
dataset = VideoCaptionDataset(segmented_data, caption_data, "/mnt/e/MML_RESEARCH/DATA", transform=transform)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# --- Step 4: Fine-Tuning Loop --- #
# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 10  # Define number of epochs
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in data_loader:
        # Move batch to device
        input_images = batch['pixel_values'].squeeze(1).to(device)  # Shape: (batch_size, 3, 224, 224)
        input_texts = batch['input_ids'].squeeze(1).to(device)  # Shape: (batch_size, sequence_length)
        attention_mask = batch['attention_mask'].squeeze(1).to(device)
        
        # Forward pass
        outputs = model(input_ids=input_texts, pixel_values=input_images, attention_mask=attention_mask)
        logits_per_image, logits_per_text = outputs.logits_per_image, outputs.logits_per_text
        
        # Compute contrastive loss (average of both directions)
        loss = (logits_per_image.mean() + logits_per_text.mean()) / 2
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data_loader)}")

# --- Step 5: Save the Fine-Tuned Model --- #
model.save_pretrained("/mnt/e/MML_RESEARCH/FINETUNEDCLIP")
print("Fine-tuning complete, model saved.")
