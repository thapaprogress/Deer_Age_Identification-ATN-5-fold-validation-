"""
HEIC to JPG Image Converter
Converts Apple HEIC format images to JPG for compatibility with PyTorch
"""

import os
from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener
import shutil
from tqdm import tqdm

# Register HEIF opener with PIL
register_heif_opener()


class HEICConverter:
    """Convert HEIC images to JPG format"""
    
    def __init__(self, source_dir, target_dir, quality=95):
        """
        Args:
            source_dir: Directory containing HEIC images
            target_dir: Directory to save converted JPG images
            quality: JPG quality (1-100)
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.quality = quality
        self.stats = {
            'total': 0,
            'converted': 0,
            'already_jpg': 0,
            'failed': 0,
            'errors': []
        }
    
    def convert_image(self, image_path, output_path):
        """
        Convert a single HEIC image to JPG
        
        Args:
            image_path: Path to HEIC image
            output_path: Path to save JPG image
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Open and convert image
            img = Image.open(image_path)
            
            # Convert RGBA to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            
            # Save as JPG
            img.save(output_path, 'JPEG', quality=self.quality)
            return True
            
        except Exception as e:
            self.stats['errors'].append({
                'file': str(image_path),
                'error': str(e)
            })
            return False
    
    def convert_directory(self, preserve_structure=True):
        """
        Convert all HEIC images in source directory to JPG
        
        Args:
            preserve_structure: If True, maintains subdirectory structure
        
        Returns:
            dict: Conversion statistics
        """
        # Find all image files
        image_extensions = {'.heic', '.jpg', '.jpeg', '.png'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.source_dir.rglob(f'*{ext}'))
            image_files.extend(self.source_dir.rglob(f'*{ext.upper()}'))
        
        self.stats['total'] = len(image_files)
        
        print(f"Found {self.stats['total']} image files")
        print(f"Converting HEIC images to JPG...")
        
        for img_path in tqdm(image_files, desc="Converting images"):
            # Calculate relative path
            rel_path = img_path.relative_to(self.source_dir)
            
            # Determine output path
            if preserve_structure:
                output_path = self.target_dir / rel_path
            else:
                output_path = self.target_dir / img_path.name
            
            # Change extension to .jpg
            output_path = output_path.with_suffix('.jpg')
            
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if already JPG
            if img_path.suffix.lower() in {'.jpg', '.jpeg'}:
                # Copy JPG files directly
                shutil.copy2(img_path, output_path)
                self.stats['already_jpg'] += 1
            else:
                # Convert HEIC to JPG
                if self.convert_image(img_path, output_path):
                    self.stats['converted'] += 1
                else:
                    self.stats['failed'] += 1
        
        return self.stats
    
    def print_stats(self):
        """Print conversion statistics"""
        print("\n" + "="*60)
        print("CONVERSION STATISTICS")
        print("="*60)
        print(f"Total images found:     {self.stats['total']}")
        print(f"HEIC converted to JPG:  {self.stats['converted']}")
        print(f"Already JPG (copied):   {self.stats['already_jpg']}")
        print(f"Failed conversions:     {self.stats['failed']}")
        
        if self.stats['errors']:
            print(f"\nErrors encountered:")
            for error in self.stats['errors'][:10]:  # Show first 10 errors
                print(f"  - {error['file']}: {error['error']}")
            if len(self.stats['errors']) > 10:
                print(f"  ... and {len(self.stats['errors']) - 10} more errors")
        
        print("="*60)


def convert_deer_dataset(source_dir, target_dir):
    """
    Convenience function to convert deer dataset
    
    Args:
        source_dir: Source directory with HEIC images
        target_dir: Target directory for JPG images
    """
    converter = HEICConverter(source_dir, target_dir, quality=95)
    stats = converter.convert_directory(preserve_structure=True)
    converter.print_stats()
    return stats


if __name__ == "__main__":
    # Example usage
    source = r"c:\Users\PRAJNA WORLD TECH\OneDrive\Desktop\atn\deer data"
    target = r"c:\Users\PRAJNA WORLD TECH\OneDrive\Desktop\atn\data\raw"
    
    print("Starting HEIC to JPG conversion...")
    print(f"Source: {source}")
    print(f"Target: {target}")
    print()
    
    convert_deer_dataset(source, target)
