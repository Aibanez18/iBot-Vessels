import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import pickle
from typing import Dict, List, Tuple
import cv2
import torchvision.transforms as T

# No need for transformers library - using torch.hub instead

class RegionProcessor:
    def __init__(self, model_name: str = "", device: str = "auto"):
        """
        Initialize the processor with DINOv2 model.
        
        Args:
            model_name: DINOv2 model variant (dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14)
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Loading DINOv2 model: {model_name} on {self.device}")
        
        # Load model
        self.model_name = model_name
        self.model_name = model_name
        if model_name == 'dinov2':
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self.model.eval().to(self.device)
            self.feature_dim = 768
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.5]*3, std=[0.5]*3)
            ])
        elif model_name == 'resnet50':
            import torchvision.models as models
            self.model = models.resnet50(pretrained=True)
            # Remove the final classification layer to get features
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval().to(self.device)
            self.feature_dim = 2048
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif model_name == "clip":
            import clip
            self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
            self.model.eval().to(self.device)
            self.feature_dim = 512
        else:
            from models.vit_lora import vit_lora
            checkpoint_path = f"checkpoints/{model_name}.pth"
            self.model = vit_lora(ckpt_path=checkpoint_path).to(self.device)              
            self.model.eval().to(self.device)
            self.feature_dim = 768
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        
        print(f"Model loaded successfully. Feature dimension: {self.feature_dim}")
    
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
    
    def extract_features(self, region: np.ndarray) -> np.ndarray:
        """
        Extract DINOv2 class token features from a region.
        
        Args:
            region: Image region as numpy array
            
        Returns:
            Class token features as numpy array
        """
        # Convert to PIL Image if needed
        if isinstance(region, np.ndarray):
            if len(region.shape) == 3:
                pil_image = Image.fromarray(region)
            else:
                pil_image = Image.fromarray(region).convert('RGB')
        else:
            pil_image = region
        
        if self.model_name == "clip":
            # Apply CLIP preprocessing
            input_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

            # Extract features
            with torch.no_grad():
                # Get image features
                image_features = self.model.encode_image(input_tensor)
                # Normalize features (CLIP typically uses normalized features)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                features = image_features.cpu().numpy()
        
            return features.squeeze()
        else:
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
                          json_path: str,
                          output_dir: str,
                          thumbnail_size: Tuple[int, int] = (224, 224),
                          save_thumbnails: bool = True) -> Dict:
        """
        Process all images from JSON data.
        
        Args:
            image_folder: Path to folder containing images
            json_path: Path to JSON file
            output_dir: Directory to save results
            thumbnail_size: Size for thumbnails
            save_thumbnails: Whether to save thumbnail images
            max_region_area: Maximum region area in pixels (absolute threshold)
            max_region_ratio: Maximum region area as ratio of total image area (0.0-1.0)
            
        Returns:
            Dictionary with all processed results
        """
        # Load JSON data
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_results = {
            'processed_images': {},
            'total_regions': 0,
            'filtered_regions': 0,
            'features_shape': None,
            'thumbnail_size': thumbnail_size
        }
        
        print(f"Found {len(json_data)} images to process...")

        for image_data in json_data['images']:
            image_filename = image_data['file_name']
            image_id = image_data['id']

            print(f"\nProcessing {image_filename}...")
            
            # Find the actual image file
            image_path = Path(image_folder) / image_filename
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue
            
            # Create output directory for this image
            image_output_dir = output_path / Path(image_filename).stem
            
            # Process this image's regions
            image_regions = [x for x in json_data['annotations'] if x['image_id'] == image_id]
            result = self.process_single_image(
                image_path=str(image_path),
                regions_data=image_regions,
                output_dir=str(image_output_dir),
                thumbnail_size=thumbnail_size,
                save_thumbnails=save_thumbnails
            )
            
            all_results['processed_images'][image_filename] = result
            all_results['total_regions'] += len(result['regions'])
            if all_results['features_shape'] is None:
                all_results['features_shape'] = result['features_shape']
        
        # Save combined results
        combined_features_path = output_path / f"all_features_{self.model_name}.pkl"
        with open(combined_features_path, 'wb') as f:
            pickle.dump(all_results, f)
        
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
                            save_thumbnails: bool = True) -> Dict:
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
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if save_thumbnails:
            thumbnail_dir = output_path / "thumbnails"
            thumbnail_dir.mkdir(exist_ok=True)
        
        results = {
            'image_path': image_path,
            'output_dir': output_dir,
            'image_dimensions': (image.shape[1], image.shape[0]),  # (width, height)
            'image_area': image_area,
            'regions': [],
            'features_shape': None,
            'thumbnail_size': thumbnail_size
        }
        
        print(f"Image dimensions: {results['image_dimensions']}, area: {image_area:,} pixels")
        
        # Process each region
        for i, region_data in enumerate(regions_data):
            x = int(region_data.get('x', region_data.get('bbox', [0])[0]))
            y = int(region_data.get('y', region_data.get('bbox', [0, 0])[1]))
            width = int(region_data.get('width', region_data.get('bbox', [0, 0, 100])[2]))
            height = int(region_data.get('height', region_data.get('bbox', [0, 0, 0, 100])[3]))
            region_area = width * height
            pattern_id = region_data['category_id']
            try:
                # Extract region
                region = self.extract_region(image, x, y, width, height)

                # Extract features
                features = self.extract_features(region)
                
                # Save thumbnail if requested
                thumbnail_path = None
                if save_thumbnails:  
                    thumbnail = self.create_thumbnail(region, thumbnail_size)
                    thumbnail_path = thumbnail_dir / f"region_{i:04d}.png"
                    Image.fromarray(thumbnail).save(thumbnail_path)
                
                # Store results
                region_result = {
                    'region_id': i,
                    'bbox': [x, y, width, height],
                    'region_area': region_area,
                    'region_ratio': region_area / image_area,
                    'features': features,
                    'pattern_id' : pattern_id, 
                    'thumbnail_path': str(thumbnail_path) if thumbnail_path else None,
                    'original_data': region_data
                }
                
                results['regions'].append(region_result)
                results['features_shape'] = features.shape
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(regions_data)} regions")
                    
            except Exception as e:
                print(f"Error processing region {i}: {e}")
                continue
        
        print(f"Processing complete! Results saved to {output_path}")
        print(f"Processed {len(results['regions'])} regions successfully")
        
        return results

# Example usage function
def process_all_images(image_folder: str, 
                          json_path: str, 
                          output_dir: str,
                          model_name: str = "dinov2_vitb14"):
    """
    Main function to process all images from JSON file.
    
    Args:
        image_folder: Path to folder containing images
        sam_json_path: Path to SAM JSON file
        output_dir: Directory to save results
        model_name: DINOv2 model variant (dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14)
    """
    # Initialize processor
    processor = RegionProcessor(model_name=model_name)
    
    # Process all images
    results = processor.process_all_images(
        image_folder=image_folder,
        json_path=json_path,
        output_dir=output_dir,
        thumbnail_size=(224, 224),
        save_thumbnails=True
    )
    
    return results

# Example usage
if __name__ == "__main__":
    # Process all images from SAM JSON
    image_folder = "D:/Universidad/Proyecto de Titulo/sam-dino_detector/flat_images"
    json_path = "D:/Universidad/Proyecto de Titulo/sam-dino_detector/annotations.json"
    output_dir = "D:/Universidad/Proyecto de Titulo/sam-dino_detector/annotated_output"
    
    # Process all images
    all_results = process_all_images(
        image_folder=image_folder,
        json_path=json_path,
        output_dir=output_dir,
        model_name="ibot_horae_lora_7e-3_30"
    )
    
    print(f"Processed {len(all_results['processed_images'])} images")
    print(f"Total regions: {all_results['total_regions']}")
    print(f"Feature dimension: {all_results['features_shape']}")