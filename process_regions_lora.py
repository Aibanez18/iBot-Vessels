import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional
import torchvision.transforms as T
import cv2

from models.vit_lora import vit_lora

# No need for transformers library - using torch.hub instead

class SAMRegionProcessor:
    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "auto", checkpoint_path: Optional[str] = None):
        """
        Initialize the processor with DINOv2 model.
        
        Args:
            model_name: DINOv2 model variant (dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14)
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model: {model_name} on {self.device}")
        self.model_name = model_name
        # Load model from torch.hub
        self.model = vit_lora(ckpt_path=checkpoint_path).to(self.device)              
        self.model.eval()
        
        # Define transforms
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # Get feature dimension based on model
        feature_dims = {
            'dinov2_vits14': 384,
            'dinov2_vitb14': 768,
            'dinov2_vitl14': 1024,
            'dinov2_vitg14': 1536
        }
        self.feature_dim = feature_dims.get(model_name, 768)
        
        print(f"Model loaded successfully. Feature dimension: {self.feature_dim}")
    
    def load_sam_data(self, json_path: str) -> Dict:
        """Load SAM data from JSON file."""
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def extract_region(self, image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Extract region from image using bounding box coordinates.
        
        Args:
            image: Input image as numpy array
            x, y: Top-left corner coordinates
            width, height: Region dimensions
            
        Returns:
            Cropped region as numpy array
        """
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        x2 = max(x + 1, min(x + width, w))
        y2 = max(y + 1, min(y + height, h))
        
        return image[y:y2, x:x2]
    
    def create_thumbnail(self, region: np.ndarray, thumbnail_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Create a thumbnail of the region with transparent borders."""
        if region.size == 0:
            # Return a transparent thumbnail if region is empty
            return np.zeros((*thumbnail_size, 4), dtype=np.uint8)

        # Convert to PIL for better resizing
        if len(region.shape) == 3:
            if region.shape[2] == 4:
                # Already has alpha channel
                pil_image = Image.fromarray(region, 'RGBA')
            else:
                # RGB image, convert to RGBA
                pil_image = Image.fromarray(region).convert('RGBA')
        else:
            # Grayscale image, convert to RGBA
            pil_image = Image.fromarray(region).convert('RGBA')

        # Resize maintaining aspect ratio
        pil_image.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)

        # Create a new RGBA image with transparent background
        thumbnail = Image.new('RGBA', thumbnail_size, (0, 0, 0, 0))
        paste_x = (thumbnail_size[0] - pil_image.width) // 2
        paste_y = (thumbnail_size[1] - pil_image.height) // 2
        thumbnail.paste(pil_image, (paste_x, paste_y))

        return np.array(thumbnail)
    
    def extract_lora_features(self, region: np.ndarray) -> np.ndarray:
        """
        Extract LoRA features from a region.

        Args:
            region: Image region as numpy array
            
        Returns:
            Class token features as numpy array
        """
        if region.size == 0:
            # Return zero features for empty regions
            return np.zeros((self.feature_dim,), dtype=np.float32)
        
        # Convert to PIL Image if needed
        if isinstance(region, np.ndarray):
            if len(region.shape) == 3:
                pil_image = Image.fromarray(region)
            else:
                pil_image = Image.fromarray(region).convert('RGB')
        else:
            pil_image = region
        
        # Apply transforms
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            # Get class token features (CLS token)
            features = self.model(input_tensor)
            class_token = features.cpu().numpy()
        
        return class_token.squeeze()
    
    def process_all_images(self,
                          image_folder: str,
                          sam_json_path: str,
                          output_dir: str,
                          thumbnail_size: Tuple[int, int] = (224, 224),
                          save_thumbnails: bool = True,
                          max_region_area: Optional[int] = None,
                          max_region_ratio: Optional[float] = 0.15) -> Dict:
        """
        Process all images from SAM JSON data.
        
        Args:
            image_folder: Path to folder containing images
            sam_json_path: Path to SAM JSON file
            output_dir: Directory to save results
            thumbnail_size: Size for thumbnails
            save_thumbnails: Whether to save thumbnail images
            max_region_area: Maximum region area in pixels (absolute threshold)
            max_region_ratio: Maximum region area as ratio of total image area (0.0-1.0)
            
        Returns:
            Dictionary with all processed results
        """
        # Load SAM data
        with open(sam_json_path, 'r') as f:
            sam_data = json.load(f)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_results = {
            'processed_images': {},
            'total_regions': 0,
            'filtered_regions': 0,
            'features_shape': None,
            'thumbnail_size': thumbnail_size,
            'filtering_params': {
                'max_region_area': max_region_area,
                'max_region_ratio': max_region_ratio
            }
        }
        
        print(f"Found {len(sam_data)} images to process...")
        
        for image_filename, image_data in sam_data.items():
            print(f"\nProcessing {image_filename}...")
            
            # Find the actual image file
            image_path = Path(image_folder) / image_filename
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue
            
            # Create output directory for this image
            image_output_dir = output_path / Path(image_filename).stem
            
            # Process this image's regions
            result = self.process_single_image(
                image_path=str(image_path),
                regions_data=image_data['regions'],
                output_dir=str(image_output_dir),
                thumbnail_size=thumbnail_size,
                save_thumbnails=save_thumbnails,
                max_region_area=max_region_area,
                max_region_ratio=max_region_ratio
            )
            
            all_results['processed_images'][image_filename] = result
            all_results['total_regions'] += len(result['regions'])
            all_results['filtered_regions'] += result.get('filtered_count', 0)
            if all_results['features_shape'] is None:
                all_results['features_shape'] = result['features_shape']
        
        # Save combined results
        combined_features_path = output_path / f"all_features_{self.model_name}.pkl"
        with open(combined_features_path, 'wb') as f:
            pickle.dump(all_results, f)
        
        # Save combined metadata
        combined_metadata = {
            'total_images': len(all_results['processed_images']),
            'total_regions': all_results['total_regions'],
            'filtered_regions': all_results['filtered_regions'],
            'features_shape': all_results['features_shape'],
            'thumbnail_size': thumbnail_size,
            'filtering_params': all_results['filtering_params'],
            'images': {}
        }
        
        for img_name, img_result in all_results['processed_images'].items():
            combined_metadata['images'][img_name] = {
                'num_regions': len(img_result['regions']),
                'output_dir': img_result.get('output_dir', ''),
                'regions': [
                    {
                        'region_id': r['region_id'],
                        'bbox': r['bbox'],
                        'thumbnail_path': r['thumbnail_path']
                    }
                    for r in img_result['regions']
                ]
            }
        
        with open(output_path / "combined_metadata.json", 'w') as f:
            json.dump(combined_metadata, f, indent=2)
        
        print("\nAll processing complete!")
        print(f"Processed {len(all_results['processed_images'])} images")
        print(f"Total regions processed: {all_results['total_regions']}")
        print(f"Total regions filtered out: {all_results['filtered_regions']}")
        print(f"Results saved to {output_path}")
        
        return all_results

    def process_single_image(self, 
                            image_path: str, 
                            regions_data: List[Dict], 
                            output_dir: str,
                            thumbnail_size: Tuple[int, int] = (224, 224),
                            save_thumbnails: bool = False,
                            max_region_area: Optional[int] = None,
                            max_region_ratio: Optional[float] = None) -> Dict:
        """
        Process regions from a single image.
        
        Args:
            image_path: Path to the original image
            regions_data: List of region dictionaries
            output_dir: Directory to save results
            thumbnail_size: Size for thumbnails
            save_thumbnails: Whether to save thumbnail images
            max_region_area: Maximum region area in pixels (absolute threshold)
            max_region_ratio: Maximum region area as ratio of total image area (0.0-1.0)
            
        Returns:
            Dictionary with processed results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Calculate image area for ratio-based filtering
        image_area = (image.shape[0] * image.shape[1])//2
        
        results = {
            'image_path': image_path,
            'output_dir': output_dir,
            'image_dimensions': (image.shape[1], image.shape[0]),  # (width, height)
            'image_area': image_area,
            'regions': [],
            'filtered_count': 0,
            'features_shape': None,
            'thumbnail_size': thumbnail_size
        }
        
        print(f"Image dimensions: {results['image_dimensions']}, area: {image_area:,} pixels")
        
        # Filter and process each region
        filtered_regions = []
        for region_data in regions_data:
            # Extract coordinates
            x = int(region_data.get('x', region_data.get('bbox', [0])[0]))
            y = int(region_data.get('y', region_data.get('bbox', [0, 0])[1]))
            width = int(region_data.get('width', region_data.get('bbox', [0, 0, 100])[2]))
            height = int(region_data.get('height', region_data.get('bbox', [0, 0, 0, 100])[3]))

            region_area = width * height
            region_ratio = region_area / (image_area)

            # Calculate aspect ratio (width/height)
            aspect_ratio = width / height

            # Apply basic filters
            should_filter = False

            if max_region_area is not None and region_area > max_region_area:
                should_filter = True

            if max_region_ratio is not None and region_ratio > max_region_ratio:
                should_filter = True

            # Filter out funky aspect ratios
            # Adjust these thresholds based on what you consider "funky"
            min_aspect_ratio = 0.135  # Very tall/thin regions
            max_aspect_ratio = 7.5  # Very wide/short regions

            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                should_filter = True

            # Filter out very small regions (likely noise)
            min_area = 100  # Adjust based on your image resolution
            if region_area < min_area:
                should_filter = True

            if should_filter:
                results['filtered_count'] += 1
                continue

            # Add area to the region data for sorting
            region_data['area'] = region_area
            filtered_regions.append(region_data)
        
        # Sort by area (largest first) and keep top half
        filtered_regions.sort(key=lambda x: x['area'], reverse=True)
        num_to_keep = len(filtered_regions) * 2 // 3  # Keep largest half
        filtered_regions = filtered_regions[:num_to_keep]
        
        print(f"Original regions: {len(regions_data)}")
        print(f"After filtering: {len(filtered_regions)}")
        
        # Process each remaining region
        for i, region_data in enumerate(filtered_regions):
            x = region_data['x']
            y = region_data['y']
            width = region_data['width']
            height = region_data['height']
            region_area = region_data['area']
            try:
                # Extract region
                region = self.extract_region(image, x, y, width, height)
                
                # Create thumbnail
                thumbnail = self.create_thumbnail(region, thumbnail_size)
                
                # Extract LoRA features
                features = self.extract_lora_features(region)
                
                # Save thumbnail if requested
                thumbnail_path = None
                if save_thumbnails:
                    thumbnail_path = thumbnail_dir / f"region_{i:04d}.png"
                    Image.fromarray(thumbnail).save(thumbnail_path)
                
                # Store results
                region_result = {
                    'region_id': i,
                    'bbox': [x, y, width, height],
                    'region_area': region_area,
                    'region_ratio': region_area / image_area,
                    'features': features,
                    'thumbnail_path': str(thumbnail_path) if thumbnail_path else None,
                    'original_data': region_data
                }
                
                results['regions'].append(region_result)
                results['features_shape'] = features.shape
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(filtered_regions)} regions")
                    
            except Exception as e:
                print(f"Error processing region {i}: {e}")
                continue
        
        return results

# Example usage function
def process_all_sam_images(image_folder: str, 
                          sam_json_path: str, 
                          output_dir: str,
                          model_name: str = "dinov2_vitb14",
                          checkpoint_path: Optional[str] = None) -> Dict:
    """
    Main function to process all images from SAM JSON file.
    
    Args:
        image_folder: Path to folder containing images
        sam_json_path: Path to SAM JSON file
        output_dir: Directory to save results
        model_name: DINOv2 model variant (dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14)
    """
    # Initialize processor
    processor = SAMRegionProcessor(model_name=model_name, checkpoint_path=checkpoint_path)
    
    # Process all images
    results = processor.process_all_images(
        image_folder=image_folder,
        sam_json_path=sam_json_path,
        output_dir=output_dir,
        thumbnail_size=(224, 224),
        save_thumbnails=False,
        max_region_ratio=0.20
    )
    
    return results

# Example usage
if __name__ == "__main__":
    # Process all images from SAM JSON
    image_folder = "eval/flat_textures/"
    sam_json_path = "eval/detected_regions_2500.json"
    output_dir = "eval/features/"
    
    # Process all images
    all_results = process_all_sam_images(
        image_folder=image_folder,
        sam_json_path=sam_json_path,
        output_dir=output_dir,
        model_name="ibot_horae_lora_7e-3_45",  # or ibot_lora_1e-3 or ibot_lora_75e-5
        checkpoint_path="checkpoints/ibot_horae_lora_7e-3_45.pth"
    )
    
    print(f"Processed {len(all_results['processed_images'])} images")
    print(f"Total regions: {all_results['total_regions']}")
    print(f"Feature dimension: {all_results['features_shape']}")